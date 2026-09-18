#include "ply_reader.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace dpe {
namespace eth3d {
namespace {

enum class PlyFormat {
    Ascii,
    BinaryLittleEndian,
};

enum class ScalarType {
    Int8,
    UInt8,
    Int16,
    UInt16,
    Int32,
    UInt32,
    Float32,
    Float64,
};

struct Property {
    ScalarType type;
    std::string name;
};

struct Header {
    PlyFormat format = PlyFormat::Ascii;
    std::size_t vertex_count = 0;
    std::vector<Property> vertex_properties;
};

ScalarType ParseScalarType(const std::string& name) {
    if (name == "char" || name == "int8") return ScalarType::Int8;
    if (name == "uchar" || name == "uint8") return ScalarType::UInt8;
    if (name == "short" || name == "int16") return ScalarType::Int16;
    if (name == "ushort" || name == "uint16") return ScalarType::UInt16;
    if (name == "int" || name == "int32") return ScalarType::Int32;
    if (name == "uint" || name == "uint32") return ScalarType::UInt32;
    if (name == "float" || name == "float32") return ScalarType::Float32;
    if (name == "double" || name == "float64") return ScalarType::Float64;
    throw std::runtime_error("Unsupported PLY scalar type: " + name);
}

Header ReadHeader(std::ifstream& input) {
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("Not a PLY file");
    }
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line != "ply") {
        throw std::runtime_error("Not a PLY file");
    }

    Header header;
    std::string current_element;
    bool format_seen = false;
    bool vertex_seen = false;
    bool preceding_nonempty_element = false;
    bool end_seen = false;

    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "end_header") {
            end_seen = true;
            break;
        }

        std::istringstream tokens(line);
        std::string keyword;
        tokens >> keyword;
        if (keyword == "format") {
            std::string format;
            tokens >> format;
            if (format == "ascii") header.format = PlyFormat::Ascii;
            else if (format == "binary_little_endian") header.format = PlyFormat::BinaryLittleEndian;
            else throw std::runtime_error("Unsupported PLY format: " + format);
            format_seen = true;
        } else if (keyword == "element") {
            std::size_t count = 0;
            tokens >> current_element >> count;
            if (current_element == "vertex") {
                if (preceding_nonempty_element) {
                    throw std::runtime_error("PLY vertex data must be the first non-empty element");
                }
                header.vertex_count = count;
                vertex_seen = true;
            } else if (!vertex_seen && count != 0) {
                preceding_nonempty_element = true;
            }
        } else if (keyword == "property" && current_element == "vertex") {
            std::string type_name;
            tokens >> type_name;
            if (type_name == "list") {
                throw std::runtime_error("List-valued vertex properties are not supported");
            }
            Property property;
            property.type = ParseScalarType(type_name);
            tokens >> property.name;
            header.vertex_properties.push_back(property);
        }
    }

    if (!end_seen || !format_seen || !vertex_seen) {
        throw std::runtime_error("Incomplete PLY header");
    }
    if (header.vertex_properties.empty()) {
        throw std::runtime_error("PLY vertex element has no properties");
    }
    return header;
}

template <typename T>
double ReadBinaryScalar(std::ifstream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) throw std::runtime_error("Unexpected end of binary PLY vertex data");
    return static_cast<double>(value);
}

double ReadBinaryValue(std::ifstream& input, ScalarType type) {
    switch (type) {
        case ScalarType::Int8: return ReadBinaryScalar<std::int8_t>(input);
        case ScalarType::UInt8: return ReadBinaryScalar<std::uint8_t>(input);
        case ScalarType::Int16: return ReadBinaryScalar<std::int16_t>(input);
        case ScalarType::UInt16: return ReadBinaryScalar<std::uint16_t>(input);
        case ScalarType::Int32: return ReadBinaryScalar<std::int32_t>(input);
        case ScalarType::UInt32: return ReadBinaryScalar<std::uint32_t>(input);
        case ScalarType::Float32: return ReadBinaryScalar<float>(input);
        case ScalarType::Float64: return ReadBinaryScalar<double>(input);
    }
    throw std::runtime_error("Unknown PLY scalar type");
}

std::uint8_t ToColor(double value) {
    value = std::max(0.0, std::min(255.0, value));
    return static_cast<std::uint8_t>(value + 0.5);
}

void AssignProperty(EvaluationPoint* point, const std::string& name, double value) {
    if (name == "x") point->x = static_cast<float>(value);
    else if (name == "y") point->y = static_cast<float>(value);
    else if (name == "z") point->z = static_cast<float>(value);
    else if (name == "red") point->red = ToColor(value);
    else if (name == "green") point->green = ToColor(value);
    else if (name == "blue") point->blue = ToColor(value);
}

void ValidateRequiredProperties(const Header& header) {
    bool x = false, y = false, z = false, red = false, green = false, blue = false;
    for (const Property& property : header.vertex_properties) {
        x = x || property.name == "x";
        y = y || property.name == "y";
        z = z || property.name == "z";
        red = red || property.name == "red";
        green = green || property.name == "green";
        blue = blue || property.name == "blue";
    }
    if (!(x && y && z && red && green && blue)) {
        throw std::runtime_error("PLY must contain x, y, z, red, green, and blue vertex properties");
    }
}

}  // namespace

std::vector<EvaluationPoint> ReadEvaluationPly(const boost::filesystem::path& path) {
    std::ifstream input(path.string(), std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open evaluation PLY: " + path.string());

    const Header header = ReadHeader(input);
    ValidateRequiredProperties(header);
    std::vector<EvaluationPoint> points(header.vertex_count);

    for (std::size_t index = 0; index < header.vertex_count; ++index) {
        EvaluationPoint point;
        for (const Property& property : header.vertex_properties) {
            double value = 0.0;
            if (header.format == PlyFormat::Ascii) {
                input >> value;
                if (!input) throw std::runtime_error("Unexpected end of ASCII PLY vertex data");
            } else {
                value = ReadBinaryValue(input, property.type);
            }
            AssignProperty(&point, property.name, value);
        }
        points[index] = point;
    }
    return points;
}

}  // namespace eth3d
}  // namespace dpe
