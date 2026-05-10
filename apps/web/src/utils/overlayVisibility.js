export function pointGroupMatches(group, layers) {
  if (group === "eyes") return Boolean(layers.eyes);
  if (group === "nose") return Boolean(layers.nose);
  if (group === "mouth") return Boolean(layers.mouth);
  if (group === "chin_jaw") return Boolean(layers.chinJaw);
  if (group === "brows") return Boolean(layers.thirds);
  if (group === "midline") return Boolean(layers.symmetry);
  return true;
}

export function segmentLayerVisible(segment, layers) {
  const segmentLayers = Array.isArray(segment?.layers) ? segment.layers : [];
  for (const layer of segmentLayers) {
    if (layer === "measurements" && !layers.measurements) return false;
    if (layer === "angles" && !layers.angles) return false;
    if (layer === "symmetry" && !layers.symmetry) return false;
    if (layer === "thirds" && !layers.thirds) return false;
    if (layer === "eyes" && !layers.eyes) return false;
    if (layer === "nose" && !layers.nose) return false;
    if (layer === "mouth" && !layers.mouth) return false;
    if (layer === "chin_jaw" && !layers.chinJaw) return false;
    if (layer === "brows" && !layers.thirds) return false;
  }
  return true;
}
