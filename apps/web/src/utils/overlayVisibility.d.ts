export type OverlayLayers = {
  measurements: boolean;
  angles: boolean;
  symmetry: boolean;
  faceOval: boolean;
  thirds: boolean;
  eyes: boolean;
  brows: boolean;
  nose: boolean;
  mouth: boolean;
  cheekbones: boolean;
  chinJaw: boolean;
};

export type OverlaySegmentLike = {
  layers?: string[];
};

export function pointGroupMatches(group: string, layers: OverlayLayers): boolean;
export function segmentLayerVisible(segment: OverlaySegmentLike, layers: OverlayLayers): boolean;
