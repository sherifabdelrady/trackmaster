"""TrackMaster — Multi-Object Tracking (DeepSORT + YOLOv8)"""
import numpy as np, cv2, argparse
from collections import defaultdict

class KalmanTracker:
    """Simplified Kalman filter for 2D bounding box tracking."""
    def __init__(self, bbox):
        self.id = None
        self.hits = 1
        self.no_detection_count = 0
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        self.state = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=float)
    def predict(self):
        self.state[:4] += self.state[4:]
        self.no_detection_count += 1
    def update(self, bbox):
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
        self.state[:4] = [cx, cy, w, h]; self.hits += 1; self.no_detection_count = 0
    def to_bbox(self):
        cx, cy, w, h = self.state[:4]
        return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]

class SORT:
    def __init__(self, max_age=3, min_hits=2, iou_thresh=0.3):
        self.trackers, self.next_id = [], 1
        self.max_age, self.min_hits, self.iou_thresh = max_age, min_hits, iou_thresh
    def iou(self, b1, b2):
        xi1,yi1,xi2,yi2 = max(b1[0],b2[0]),max(b1[1],b2[1]),min(b1[2],b2[2]),min(b1[3],b2[3])
        inter = max(0,xi2-xi1)*max(0,yi2-yi1)
        a1=(b1[2]-b1[0])*(b1[3]-b1[1]); a2=(b2[2]-b2[0])*(b2[3]-b2[1])
        return inter/(a1+a2-inter+1e-6)
    def update(self, detections):
        for t in self.trackers: t.predict()
        if not detections:
            self.trackers = [t for t in self.trackers if t.no_detection_count <= self.max_age]
            return []
        matched, unmatched_d = set(), []
        for di, d in enumerate(detections):
            best_iou, best_t = 0, -1
            for ti, t in enumerate(self.trackers):
                iou = self.iou(d, t.to_bbox())
                if iou > best_iou: best_iou, best_t = iou, ti
            if best_iou >= self.iou_thresh and best_t not in matched:
                self.trackers[best_t].update(d); matched.add(best_t)
            else:
                unmatched_d.append(d)
        for d in unmatched_d:
            t = KalmanTracker(d); t.id = self.next_id; self.next_id += 1; self.trackers.append(t)
        self.trackers = [t for t in self.trackers if t.no_detection_count <= self.max_age]
        return [(t.id, t.to_bbox()) for t in self.trackers if t.hits >= self.min_hits]

if __name__ == "__main__":
    tracker = SORT(); print("SORT tracker ready. Use tracker.update(detections) per frame.")
