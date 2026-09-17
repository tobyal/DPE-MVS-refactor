from __future__ import annotations

import argparse
import asyncio
import json
import mimetypes
import struct
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


def safe_join(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    root = root.resolve()
    if root != p and root not in p.parents:
        raise HTTPException(403, "path escapes root")
    return p


class DmbFile:
    """Minimal random-access reader for the DPE/OpenCV binary Mat format."""
    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as f:
            raw = f.read(16)
        if len(raw) != 16:
            raise ValueError(f"invalid DMB header: {path}")
        self.version, self.rows, self.cols, self.cv_type = struct.unpack("<4i", raw)
        if self.version != 1:
            raise ValueError(f"unsupported DMB version {self.version}: {path}")
        self.depth = self.cv_type & 7
        self.channels = (self.cv_type >> 3) + 1
        self.formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "i", 5: "f", 6: "d"}
        if self.depth not in self.formats:
            raise ValueError(f"unsupported OpenCV depth {self.depth}: {path}")
        self.scalar_fmt = self.formats[self.depth]
        self.scalar_size = struct.calcsize("<" + self.scalar_fmt)
        self.elem_size = self.scalar_size * self.channels

    def at(self, x: int, y: int):
        if x < 0 or y < 0 or x >= self.cols or y >= self.rows:
            return None
        off = 16 + (y * self.cols + x) * self.elem_size
        with self.path.open("rb") as f:
            f.seek(off)
            raw = f.read(self.elem_size)
        if len(raw) != self.elem_size:
            return None
        vals = struct.unpack("<" + self.scalar_fmt * self.channels, raw)
        return vals[0] if self.channels == 1 else list(vals)


@lru_cache(maxsize=256)
def dmb(path_str: str) -> DmbFile:
    return DmbFile(Path(path_str))


@lru_cache(maxsize=128)
def sparse_anchors(path_str: str) -> dict[int, list[list[int]]]:
    path = Path(path_str)
    if not path.is_file():
        return {}
    out: dict[int, list[list[int]]] = {}
    with path.open("rb") as f:
        magic, count = struct.unpack("<II", f.read(8))
        if magic != 0x44504131:
            return {}
        for _ in range(count):
            pixel = struct.unpack("<i", f.read(4))[0]
            xy = struct.unpack("<16h", f.read(32))
            out[pixel] = [[int(xy[i]), int(xy[i+1])] for i in range(0,16,2) if xy[i] >= 0 and xy[i+1] >= 0]
    return out


@lru_cache(maxsize=128)
def sparse_planes(path_str: str) -> dict[int, dict[str, Any]]:
    path = Path(path_str)
    if not path.is_file():
        return {}
    out: dict[int, dict[str, Any]] = {}
    with path.open("rb") as f:
        magic, count = struct.unpack("<II", f.read(8))
        if magic != 0x44505031:
            return {}
        for _ in range(count):
            pixel = struct.unpack("<i", f.read(4))[0]
            plane = list(struct.unpack("<4f", f.read(16)))
            radius = struct.unpack("<i", f.read(4))[0]
            out[pixel] = {"plane": plane, "radius": radius}
    return out


class StudioIndex:
    def __init__(self, experiment_root: Path, dense_root: Path | None):
        self.experiment_root = experiment_root.resolve()
        self.dense_root = dense_root.resolve() if dense_root else None

    def cases(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.experiment_root.exists():
            return rows
        for p in sorted(self.experiment_root.iterdir()):
            mf = p / "manifest.json"
            if not mf.is_file():
                continue
            try:
                data = json.loads(mf.read_text())
                rows.append({"name": data.get("name", p.name), "description": data.get("description", ""),
                             "status": data.get("status", "unknown"), "views": len(data.get("views", [])),
                             "telemetry": bool(data.get("telemetry", False))})
            except Exception as exc:
                rows.append({"name": p.name, "description": f"manifest error: {exc}", "status": "error", "views": 0, "telemetry": False})
        return rows

    def manifest(self, case: str) -> dict[str, Any]:
        p = safe_join(self.experiment_root, case) / "manifest.json"
        if not p.is_file():
            raise HTTPException(404, f"manifest not found for {case}")
        data = json.loads(p.read_text())
        stat = p.stat()
        data["cache_token"] = f"{stat.st_mtime_ns}-{stat.st_size}"
        return data

    def dense_path(self, manifest: dict[str, Any]) -> Path:
        if self.dense_root:
            return self.dense_root
        raw = manifest.get("dense_root")
        if not raw:
            raise HTTPException(404, "dense root not configured")
        return Path(raw).resolve()


def make_app(experiment_root: Path, dense_root: Path | None, frontend_dist: Path | None) -> FastAPI:
    app = FastAPI(title="DPE Studio", version="0.2")
    idx = StudioIndex(experiment_root, dense_root)

    @app.get("/api/health")
    def health():
        return {"ok": True, "experiment_root": str(idx.experiment_root), "dense_root": str(idx.dense_root) if idx.dense_root else None}

    @app.get("/api/cases")
    def cases(): return idx.cases()

    @app.get("/api/cases/{case}/manifest")
    def manifest(case: str):
        return JSONResponse(idx.manifest(case), headers={"Cache-Control": "no-store"})

    def artifact_response(path: Path, media_type: str, version: str | None, filename: str | None = None):
        cache_control = "public, max-age=31536000, immutable" if version else "no-cache"
        return FileResponse(
            path,
            media_type=media_type,
            filename=filename,
            headers={"Cache-Control": cache_control},
        )

    @app.get("/api/cases/{case}/summary")
    def summary(case: str):
        p = safe_join(idx.experiment_root, case) / "telemetry_summary.json"
        if not p.is_file(): return {"views": []}
        return json.loads(p.read_text())

    @app.get("/api/cases/{case}/point-cloud")
    def point_cloud(case: str, version: str | None = Query(None, alias="v")):
        mf = idx.manifest(case); case_dir = safe_join(idx.experiment_root, case)
        p = safe_join(case_dir, mf.get("point_cloud", "DPE.ply"))
        if not p.is_file(): raise HTTPException(404, "point cloud not found")
        return artifact_response(p, "application/octet-stream", version, p.name)

    @app.get("/api/cases/{case}/gt-point-cloud")
    def gt_point_cloud(case: str, version: str | None = Query(None, alias="v")):
        mf = idx.manifest(case); rel = mf.get("ground_truth_point_cloud")
        if not rel: raise HTTPException(404, "ground truth point cloud not configured")
        p = safe_join(idx.experiment_root, str(Path(case) / rel))
        if not p.is_file(): raise HTTPException(404, "ground truth point cloud not found")
        return artifact_response(p, "application/octet-stream", version, p.name)

    @app.get("/api/cases/{case}/views/{view_id}/image")
    def image(case: str, view_id: int, version: str | None = Query(None, alias="v")):
        mf = idx.manifest(case); view = next((v for v in mf.get("views", []) if int(v.get("id", -1)) == view_id), None)
        if view is None: raise HTTPException(404, "view not found")
        p = safe_join(idx.dense_path(mf), view["image"])
        if not p.is_file(): raise HTTPException(404, "image not found")
        return artifact_response(p, mimetypes.guess_type(str(p))[0] or "image/jpeg", version)

    @app.get("/api/cases/{case}/views/{view_id}/layer/{layer}")
    def layer(case: str, view_id: int, layer: str, version: str | None = Query(None, alias="v")):
        allowed = {
            "depth":"depth.jpg", "normal":"normal.jpg", "state":"state.jpg",
            "gt_depth":"gt_depth.jpg", "gt_normal":"gt_normal.jpg", "gt_geometry_edge":"gt_geometry_edge.png", "edge_relation":"edge_relation.png",
            "gt_surface_label":"gt_surface_label.png", "fine_edge":"fine_edge.png", "coarse_region":"coarse_region.png",
            "texture_complexity":"texture_complexity.png", "es_candidate_count":"es_candidate_count.png", "es_same_ratio":"es_same_ratio.png",
            "candidate_count":"candidate_count.png",
            "candidate_same_ratio":"candidate_same_ratio.png", "anchor_same_ratio":"anchor_same_ratio.png",
            "plane_depth_error":"plane_depth_error.png", "plane_normal_error":"plane_normal_error.png",
            "radius_violation":"radius_violation.png", "final_depth_error":"final_depth_error.png",
            "matching_cost":"matching_cost.png", "adaptive_radius":"adaptive_radius.png",
        }
        if layer not in allowed: raise HTTPException(404, "unknown layer")
        p = safe_join(idx.experiment_root, case) / "views" / f"{view_id:08d}" / allowed[layer]
        if not p.is_file(): raise HTTPException(404, "layer not found")
        return artifact_response(p, mimetypes.guess_type(str(p))[0] or "image/png", version)

    @app.get("/api/cases/{case}/views/{view_id}/pixel")
    def pixel(case: str, view_id: int, x: int = Query(..., ge=0), y: int = Query(..., ge=0)):
        mf = idx.manifest(case); view = next((v for v in mf.get("views", []) if int(v.get("id", -1)) == view_id), None)
        if view is None: raise HTTPException(404, "view not found")
        w, h = int(view["width"]), int(view["height"])
        if x >= w or y >= h: raise HTTPException(400, "pixel outside image")
        base = safe_join(idx.experiment_root, case) / "views" / f"{view_id:08d}"
        def read(name: str):
            p = base / f"{name}.dmb"
            return dmb(str(p.resolve())).at(x,y) if p.is_file() else None
        result: dict[str, Any] = {"x":x,"y":y,"depth":read("depth"),"normal":read("normal"),"state":read("state"),
            "gt_depth":read("gt_depth"),"gt_normal":read("gt_normal"),"gt_valid":read("gt_valid"),
            "gt_geometry_edge":read("gt_geometry_edge"),"gt_surface_label":read("gt_surface_label"),
            "fine_edge":read("fine_edge"),"coarse_region":read("coarse_region"),"texture_complexity":read("texture_complexity"),
            "es_candidate_count":read("es_candidate_count"),"es_same_surface":read("es_same_surface"),
            "candidate_count":read("candidate_count"),"same_surface_candidates":read("same_surface_candidates"),
            "anchor_count":read("anchor_count"),"same_surface_anchors":read("same_surface_anchors"),
            "plane_depth_error":read("plane_depth_error"),"plane_normal_error":read("plane_normal_error"),
            "radius_violation":read("radius_violation"),"final_depth_error":read("final_depth_error"),
            "matching_cost":read("matching_cost"),"adaptive_radius":read("adaptive_radius")}
        pix = y*w+x
        result["anchors"] = sparse_anchors(str((base/"anchors.bin").resolve())).get(pix,[])
        result["fitted_plane"] = sparse_planes(str((base/"planes.bin").resolve())).get(pix)
        anchor_details=[]
        depth_file=base/"depth.dmb"; gt_label_file=base/"gt_surface_label.dmb"
        src_label=result.get("gt_surface_label")
        for ax,ay in result["anchors"]:
            ad=dmb(str(depth_file.resolve())).at(ax,ay) if depth_file.is_file() else None
            al=dmb(str(gt_label_file.resolve())).at(ax,ay) if gt_label_file.is_file() else None
            anchor_details.append({"x":ax,"y":ay,"depth":ad,"gt_surface_label":al,"same_surface": bool(src_label and al==src_label)})
        result["anchor_details"]=anchor_details
        return result

    @app.get("/api/status")
    def status():
        p = idx.experiment_root / "status.json"
        if not p.is_file(): return {"case":"","phase":"idle","completed":0,"total":0,"message":""}
        try: return json.loads(p.read_text())
        except Exception: return {"case":"","phase":"unknown","completed":0,"total":0,"message":"invalid status.json"}

    @app.websocket("/ws/status")
    async def ws_status(ws: WebSocket):
        await ws.accept(); last=None
        try:
            while True:
                p=idx.experiment_root/"status.json"; payload=p.read_text() if p.is_file() else '{"phase":"idle","completed":0,"total":0}'
                if payload!=last: await ws.send_text(payload); last=payload
                await asyncio.sleep(1.0)
        except WebSocketDisconnect: pass

    if frontend_dist and frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    return app


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True,type=Path); ap.add_argument("--dense",type=Path,default=None)
    ap.add_argument("--host",default="127.0.0.1"); ap.add_argument("--port",default=8765,type=int)
    ap.add_argument("--frontend",type=Path,default=Path(__file__).resolve().parents[1]/"frontend"/"dist")
    args=ap.parse_args(); import uvicorn
    uvicorn.run(make_app(args.root,args.dense,args.frontend),host=args.host,port=args.port)

if __name__=="__main__": main()
