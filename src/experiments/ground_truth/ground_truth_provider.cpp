#include "experiments/ground_truth/ground_truth_provider.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>

namespace dpe {
namespace {

struct PlyProperty {
    std::string type;
    std::string name;
};

size_t TypeSize(const std::string& t) {
    if (t == "char" || t == "int8" || t == "uchar" || t == "uint8") return 1;
    if (t == "short" || t == "int16" || t == "ushort" || t == "uint16") return 2;
    if (t == "int" || t == "int32" || t == "uint" || t == "uint32" || t == "float" || t == "float32") return 4;
    if (t == "double" || t == "float64") return 8;
    throw std::runtime_error("Unsupported PLY scalar type: " + t);
}

template <typename T>
T ReadLittle(std::istream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("Unexpected EOF while reading PLY");
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    unsigned char* b = reinterpret_cast<unsigned char*>(&value);
    std::reverse(b, b + sizeof(T));
#endif
    return value;
}

float ReadScalarAsFloat(std::istream& in, const std::string& t) {
    if (t == "char" || t == "int8") return static_cast<float>(ReadLittle<int8_t>(in));
    if (t == "uchar" || t == "uint8") return static_cast<float>(ReadLittle<uint8_t>(in));
    if (t == "short" || t == "int16") return static_cast<float>(ReadLittle<int16_t>(in));
    if (t == "ushort" || t == "uint16") return static_cast<float>(ReadLittle<uint16_t>(in));
    if (t == "int" || t == "int32") return static_cast<float>(ReadLittle<int32_t>(in));
    if (t == "uint" || t == "uint32") return static_cast<float>(ReadLittle<uint32_t>(in));
    if (t == "float" || t == "float32") return ReadLittle<float>(in);
    if (t == "double" || t == "float64") return static_cast<float>(ReadLittle<double>(in));
    throw std::runtime_error("Unsupported PLY type: " + t);
}

float DegToRad(float d) { return d * 3.14159265358979323846f / 180.0f; }

cv::Vec3f Normalize(const cv::Vec3f& n) {
    const float s = std::sqrt(n.dot(n));
    return s > 1e-10f ? n / s : cv::Vec3f(0, 0, 0);
}

float NormalAngle(const cv::Vec3f& a, const cv::Vec3f& b) {
    const float na = std::sqrt(a.dot(a));
    const float nb = std::sqrt(b.dot(b));
    if (na < 1e-8f || nb < 1e-8f) return 3.14159265358979323846f;
    float d = a.dot(b) / (na * nb);
    d = std::max(-1.0f, std::min(1.0f, d));
    return std::acos(d);
}

cv::Vec3f CamPoint(const Camera& c, int x, int y, float d) {
    return cv::Vec3f(d * (x - c.K[2]) / c.K[0], d * (y - c.K[5]) / c.K[4], d);
}

cv::Vec3f CamNormalToWorld(const Camera& c, const cv::Vec3f& n) {
    return Normalize(cv::Vec3f(
        c.R[0] * n[0] + c.R[3] * n[1] + c.R[6] * n[2],
        c.R[1] * n[0] + c.R[4] * n[1] + c.R[7] * n[2],
        c.R[2] * n[0] + c.R[5] * n[1] + c.R[8] * n[2]));
}

}  // namespace

GroundTruthProvider::GroundTruthProvider(GroundTruthConfig config) : config_(std::move(config)) {
    LoadAlignedScans();
    if (points_.empty()) throw std::runtime_error("GroundTruthProvider loaded zero scan points");
}

std::vector<GroundTruthProvider::Point3> GroundTruthProvider::ReadPlyVertices(
    const boost::filesystem::path& path) const {
    std::ifstream in(path.string(), std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open ETH3D scan PLY: " + path.string());

    std::string line;
    std::getline(in, line);
    if (line != "ply") throw std::runtime_error("Not a PLY file: " + path.string());

    enum class Format { Ascii, BinaryLittle };
    Format format = Format::Ascii;
    size_t vertex_count = 0;
    bool in_vertex = false;
    std::vector<PlyProperty> props;
    while (std::getline(in, line)) {
        if (line == "end_header") break;
        std::istringstream ls(line);
        std::string tag;
        ls >> tag;
        if (tag == "format") {
            std::string f; ls >> f;
            if (f == "ascii") format = Format::Ascii;
            else if (f == "binary_little_endian") format = Format::BinaryLittle;
            else throw std::runtime_error("Only ASCII and binary_little_endian PLY are supported: " + path.string());
        } else if (tag == "element") {
            std::string name; size_t n = 0; ls >> name >> n;
            in_vertex = (name == "vertex");
            if (in_vertex) vertex_count = n;
        } else if (tag == "property" && in_vertex) {
            std::string type, name; ls >> type;
            if (type == "list") throw std::runtime_error("List property inside PLY vertex element is unsupported");
            ls >> name;
            props.push_back({type, name});
        }
    }

    int ix = -1, iy = -1, iz = -1;
    for (size_t i = 0; i < props.size(); ++i) {
        if (props[i].name == "x") ix = static_cast<int>(i);
        if (props[i].name == "y") iy = static_cast<int>(i);
        if (props[i].name == "z") iz = static_cast<int>(i);
    }
    if (ix < 0 || iy < 0 || iz < 0) throw std::runtime_error("PLY has no x/y/z vertex properties: " + path.string());

    std::vector<Point3> out;
    out.reserve(vertex_count);
    if (format == Format::Ascii) {
        for (size_t i = 0; i < vertex_count; ++i) {
            if (!std::getline(in, line)) throw std::runtime_error("Unexpected EOF in ASCII PLY: " + path.string());
            std::istringstream ls(line);
            std::vector<float> values(props.size(), 0.0f);
            for (size_t p = 0; p < props.size(); ++p) ls >> values[p];
            out.push_back({values[ix], values[iy], values[iz]});
        }
    } else {
        for (size_t i = 0; i < vertex_count; ++i) {
            Point3 p{};
            for (size_t k = 0; k < props.size(); ++k) {
                const float v = ReadScalarAsFloat(in, props[k].type);
                if (static_cast<int>(k) == ix) p.x = v;
                else if (static_cast<int>(k) == iy) p.y = v;
                else if (static_cast<int>(k) == iz) p.z = v;
            }
            out.push_back(p);
        }
    }
    return out;
}

std::vector<std::pair<boost::filesystem::path, GroundTruthProvider::Matrix44>>
GroundTruthProvider::ParseMlp() const {
    std::vector<std::pair<boost::filesystem::path, Matrix44>> scans;
    if (config_.scan_alignment_mlp.empty() || !boost::filesystem::exists(config_.scan_alignment_mlp)) return scans;

    std::ifstream in(config_.scan_alignment_mlp.string());
    if (!in) throw std::runtime_error("Cannot open scan_alignment.mlp: " + config_.scan_alignment_mlp.string());
    std::string xml((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    const std::regex mesh_re(R"(<MLMesh[^>]*filename\s*=\s*\"([^\"]+)\"[^>]*>([\s\S]*?)</MLMesh>)", std::regex::icase);
    const std::regex matrix_re(R"(<MLMatrix44>([\s\S]*?)</MLMatrix44>)", std::regex::icase);
    for (std::sregex_iterator it(xml.begin(), xml.end(), mesh_re), end; it != end; ++it) {
        const std::string filename = (*it)[1].str();
        const std::string body = (*it)[2].str();
        std::smatch mm;
        if (!std::regex_search(body, mm, matrix_re)) continue;
        std::istringstream ms(mm[1].str());
        Matrix44 M{};
        bool ok = true;
        for (int i = 0; i < 16; ++i) if (!(ms >> M.v[i])) { ok = false; break; }
        if (!ok) continue;
        boost::filesystem::path p(filename);
        if (p.is_relative()) p = config_.scan_alignment_mlp.parent_path() / p;
        scans.push_back({p, M});
    }
    return scans;
}

void GroundTruthProvider::LoadAlignedScans() {
    const auto scans = ParseMlp();
    if (!scans.empty()) {
        for (const auto& item : scans) {
            if (!boost::filesystem::exists(item.first)) continue;
            const auto local = ReadPlyVertices(item.first);
            const Matrix44& M = item.second;
            points_.reserve(points_.size() + local.size());
            for (const auto& p : local) {
                const float x = M.v[0]*p.x + M.v[1]*p.y + M.v[2]*p.z + M.v[3];
                const float y = M.v[4]*p.x + M.v[5]*p.y + M.v[6]*p.z + M.v[7];
                const float z = M.v[8]*p.x + M.v[9]*p.y + M.v[10]*p.z + M.v[11];
                points_.push_back({x,y,z});
            }
        }
    }

    if (points_.empty()) {
        if (config_.scan_ply.empty()) throw std::runtime_error("No usable scan referenced by MLP and --gt-scan was not supplied");
        const auto local = ReadPlyVertices(config_.scan_ply);
        Matrix44 M{};
        for (int i = 0; i < 16; ++i) M.v[i] = (i % 5 == 0) ? 1.0f : 0.0f;
        // If the MLP contains an entry matching the explicitly supplied scan, apply its pose.
        for (const auto& item : scans) {
            if (item.first.filename() == config_.scan_ply.filename()) { M = item.second; break; }
        }
        points_.reserve(local.size());
        for (const auto& p : local) {
            points_.push_back({
                M.v[0]*p.x + M.v[1]*p.y + M.v[2]*p.z + M.v[3],
                M.v[4]*p.x + M.v[5]*p.y + M.v[6]*p.z + M.v[7],
                M.v[8]*p.x + M.v[9]*p.y + M.v[10]*p.z + M.v[11]});
        }
    }
}

GroundTruthFrame GroundTruthProvider::Project(const Camera& c) const {
    GroundTruthFrame out;
    out.depth = cv::Mat(c.height, c.width, CV_32F, cv::Scalar(0));
    out.normal = cv::Mat(c.height, c.width, CV_32FC3, cv::Scalar(0,0,0));
    out.valid = cv::Mat(c.height, c.width, CV_8U, cv::Scalar(0));
    out.geometry_edge = cv::Mat(c.height, c.width, CV_8U, cv::Scalar(0));
    out.surface_label = cv::Mat(c.height, c.width, CV_32S, cv::Scalar(0));

    cv::Mat zbuf(c.height, c.width, CV_32F, cv::Scalar(std::numeric_limits<float>::infinity()));
    const int r = std::max(0, config_.splat_radius);
    for (const auto& P : points_) {
        const float X = c.R[0]*P.x + c.R[1]*P.y + c.R[2]*P.z + c.t[0];
        const float Y = c.R[3]*P.x + c.R[4]*P.y + c.R[5]*P.z + c.t[1];
        const float Z = c.R[6]*P.x + c.R[7]*P.y + c.R[8]*P.z + c.t[2];
        if (!(Z > 0.0f) || !std::isfinite(Z)) continue;
        const float den = c.K[6]*X + c.K[7]*Y + c.K[8]*Z;
        if (std::fabs(den) < 1e-12f) continue;
        const float uf = (c.K[0]*X + c.K[1]*Y + c.K[2]*Z) / den;
        const float vf = (c.K[3]*X + c.K[4]*Y + c.K[5]*Z) / den;
        const int u = static_cast<int>(std::lround(uf));
        const int v = static_cast<int>(std::lround(vf));
        if (u < -r || v < -r || u >= c.width + r || v >= c.height + r) continue;
        for (int dy = -r; dy <= r; ++dy) for (int dx = -r; dx <= r; ++dx) {
            if (dx*dx + dy*dy > r*r) continue;
            const int x = u + dx, y = v + dy;
            if (x < 0 || y < 0 || x >= c.width || y >= c.height) continue;
            float& z = zbuf.at<float>(y,x);
            if (Z < z) z = Z;
        }
    }
    for (int y=0;y<c.height;++y) for (int x=0;x<c.width;++x) {
        const float z=zbuf.at<float>(y,x);
        if (std::isfinite(z)) { out.depth.at<float>(y,x)=z; out.valid.at<unsigned char>(y,x)=255; }
    }
    EstimateNormalsAndSurfaces(c, out);
    return out;
}

void GroundTruthProvider::EstimateNormalsAndSurfaces(const Camera& c, GroundTruthFrame& f) const {
    const float normal_edge = DegToRad(config_.normal_edge_degrees);
    const float surface_normal = DegToRad(config_.surface_normal_degrees);

    for (int y=1;y<c.height-1;++y) for (int x=1;x<c.width-1;++x) {
        if (!f.valid.at<unsigned char>(y,x)) continue;
        auto valid = [&](int xx,int yy){return f.valid.at<unsigned char>(yy,xx)!=0;};
        int xl=x-1,xr=x+1,yu=y-1,yd=y+1;
        if (!valid(xl,y) || !valid(xr,y) || !valid(x,yu) || !valid(x,yd)) continue;
        const cv::Vec3f L=CamPoint(c,xl,y,f.depth.at<float>(y,xl));
        const cv::Vec3f R=CamPoint(c,xr,y,f.depth.at<float>(y,xr));
        const cv::Vec3f U=CamPoint(c,x,yu,f.depth.at<float>(yu,x));
        const cv::Vec3f D=CamPoint(c,x,yd,f.depth.at<float>(yd,x));
        const cv::Vec3f dx=R-L, dy=D-U;
        cv::Vec3f n=dx.cross(dy);
        if (n[2] > 0) n = -n;  // face the reference camera
        f.normal.at<cv::Vec3f>(y,x)=CamNormalToWorld(c,n);
    }

    const int nx[4]={-1,1,0,0}, ny[4]={0,0,-1,1};
    for (int y=0;y<c.height;++y) for (int x=0;x<c.width;++x) {
        if (!f.valid.at<unsigned char>(y,x)) continue;
        const float d=f.depth.at<float>(y,x);
        const cv::Vec3f n=f.normal.at<cv::Vec3f>(y,x);
        bool edge=false;
        for(int k=0;k<4&&!edge;++k){
            const int xx=x+nx[k],yy=y+ny[k];
            if(xx<0||yy<0||xx>=c.width||yy>=c.height)continue;
            if(!f.valid.at<unsigned char>(yy,xx)) continue;
            const float d2=f.depth.at<float>(yy,xx);
            if(std::fabs(d-d2)/std::max(d,1e-6f)>config_.relative_depth_edge){edge=true;break;}
            const cv::Vec3f n2=f.normal.at<cv::Vec3f>(yy,xx);
            if(n.dot(n)>1e-8f&&n2.dot(n2)>1e-8f&&NormalAngle(n,n2)>normal_edge){edge=true;break;}
        }
        if(edge)f.geometry_edge.at<unsigned char>(y,x)=255;
    }

    int next_label=1;
    std::queue<cv::Point> q;
    for(int y=0;y<c.height;++y) for(int x=0;x<c.width;++x){
        if(!f.valid.at<unsigned char>(y,x)||f.surface_label.at<int>(y,x)!=0)continue;
        f.surface_label.at<int>(y,x)=next_label;q.push({x,y});
        while(!q.empty()){
            const cv::Point p=q.front();q.pop();
            const float d0=f.depth.at<float>(p.y,p.x);
            const cv::Vec3f n0=f.normal.at<cv::Vec3f>(p.y,p.x);
            for(int k=0;k<4;++k){
                const int xx=p.x+nx[k],yy=p.y+ny[k];
                if(xx<0||yy<0||xx>=c.width||yy>=c.height)continue;
                if(!f.valid.at<unsigned char>(yy,xx)||f.surface_label.at<int>(yy,xx)!=0)continue;
                const float d1=f.depth.at<float>(yy,xx);
                if(std::fabs(d0-d1)/std::max(d0,1e-6f)>config_.surface_depth_ratio)continue;
                const cv::Vec3f n1=f.normal.at<cv::Vec3f>(yy,xx);
                if(n0.dot(n0)>1e-8f&&n1.dot(n1)>1e-8f&&NormalAngle(n0,n1)>surface_normal)continue;
                f.surface_label.at<int>(yy,xx)=next_label;q.push({xx,yy});
            }
        }
        ++next_label;
    }
}

void GroundTruthProvider::WritePreviewPly(const boost::filesystem::path& path, size_t max_points) const {
    boost::filesystem::create_directories(path.parent_path());
    const size_t step = std::max<size_t>(1, points_.size() / std::max<size_t>(1, max_points));
    const size_t count = (points_.size() + step - 1) / step;
    std::ofstream out(path.string());
    if (!out) throw std::runtime_error("Cannot write GT preview PLY: " + path.string());
    out << "ply\nformat ascii 1.0\nelement vertex " << count << "\n"
        << "property float x\nproperty float y\nproperty float z\n"
        << "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
    for (size_t i = 0; i < points_.size(); i += step) {
        const auto& p = points_[i];
        out << p.x << ' ' << p.y << ' ' << p.z << " 245 180 70\n";
    }
}

const GroundTruthFrame& GroundTruthProvider::Frame(int image_id, const Camera& camera) {
    CacheKey key{image_id,camera.width,camera.height};
    if (has_last_ && key == last_key_) return last_frame_;
    last_key_ = key;
    last_frame_ = Project(camera);
    has_last_ = true;
    return last_frame_;
}

}  // namespace dpe
