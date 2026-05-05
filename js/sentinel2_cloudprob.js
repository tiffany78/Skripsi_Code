Map.centerObject(area2, 10);

var s2Sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED');
var s2Clouds = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY');

var START_DATE = ee.Date('2024-01-01');
var END_DATE = ee.Date('2025-01-01');
var MAX_CLOUD_PROBABILITY = 65;

function maskClouds(img) {
  var clouds = ee.Image(img.get('cloud_mask')).select('probability');
  var isNotCloud = clouds.lt(MAX_CLOUD_PROBABILITY);
  return img.updateMask(isNotCloud);
}

// The masks for the 10m bands sometimes do not exclude bad data at
// scene edges, so we apply masks from the 20m and 60m bands as well.
// Example asset that needs this operation:
// COPERNICUS/S2_CLOUD_PROBABILITY/20190301T000239_20190301T000238_T55GDP
function maskEdges(s2_img) {
  return s2_img.updateMask(
      s2_img.select('B8A').mask().updateMask(s2_img.select('B9').mask()));
}

// Filter input collections by desired data range and area2.
var criteria = ee.Filter.and(
    ee.Filter.bounds(area2), ee.Filter.date(START_DATE, END_DATE));
s2Sr = s2Sr.filter(criteria).map(maskEdges);
s2Clouds = s2Clouds.filter(criteria);

// Join S2 SR with cloud probability dataset to add cloud mask.
var s2SrWithCloudMask = ee.Join.saveFirst('cloud_mask').apply({
  primary: s2Sr,
  secondary: s2Clouds,
  condition:
      ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
});

var s2CloudMasked =
    ee.ImageCollection(s2SrWithCloudMask).map(maskClouds).median();
var rgbVis = {min: 0, max: 3000, bands: ['B4', 'B3', 'B2']};

Map.addLayer(
    s2CloudMasked.clip(area2), rgbVis, 'S2 SR masked at ' + MAX_CLOUD_PROBABILITY + '%',
    true);

var bands = [
'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12'
];
var imageConvert = s2CloudMasked.select(bands);
var lonLat = ee.Image.pixelLonLat();
var imageWithCoord = imageConvert.addBands(lonLat);
Export.image.toDrive({
  image: imageWithCoord,
  description: "Sentinel2_65CloudProb_10_2024",
  folder: 'TA_GEOTIFF',
  scale: 10,
  region: area2,
  maxPixels: 1e13,
  crs: 'EPSG:4326',
  fileFormat: "GeoTIFF"
});