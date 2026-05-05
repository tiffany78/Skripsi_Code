// Asset Asmat
var batas = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/gadm41_IDN_2");
var asmat = batas.filter(
  ee.Filter.and(
    ee.Filter.eq('NAME_1', 'Papua'),
    ee.Filter.eq('NAME_2', 'Asmat')
  )
);
Map.addLayer(asmat, {color: 'gray', strokeWidth: 2}, 'Asmat');
Map.centerObject(asmat, 8);

// Pemanggilan Asset
var batas50K = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Garis50K");
// Pembuatan Layer
// Map.addLayer(batas50K, {color: 'yellow', strokeWidth : 10},'batas50K');

// GARIS PANTAI
var garisPantaiAsmat = batas50K.filterBounds(asmat);
var coastlineAll = garisPantaiAsmat.geometry().dissolve();
var coastline = coastlineAll.intersection(
  asmat.geometry().buffer(500),
  ee.ErrorMargin(1)
);
// Map.addLayer(coastline, {color:'red'}, 'Coastline Asmat FIX');

var panjang = coastline.length();
print('Panjang garis (m):', panjang);
var interval = 1000; // 1 km
var distances = ee.List.sequence(0, panjang, interval);
var segments = coastline.cutLines(distances);

var titikSampling = ee.FeatureCollection(
  segments.geometries().map(function(g) {
    return ee.Feature(ee.Geometry(g).centroid());
  })
);
Map.addLayer(titikSampling, {color:'red'}, 'Titik 1km');

// ROI BUFFER
var jarak = 24 * 1852; // meter
var buffer = titikSampling.map(function(f) {
  return f.buffer(jarak);
});
Map.addLayer(buffer, {color:'green'}, 'Buffer 12 Mil');

// VISUALUSASI CITRA
var csPlus=ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var composite = s2
    .filterDate("2025-01-01", "2025-12-31")
    .filterBounds(buffer)
    .linkCollection(csPlus, [QA_BAND])
    .map(function(img) {
      return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));})
    ;
composite = composite.median();

Map.addLayer(
    composite.clip(area), 
    {bands: ['B4','B3','B2'], min:0, max:2500}, 
    'RGB Median');
Map.addLayer(batas50K, {color: 'yellow', strokeWidth : 10},'batas50K');

// EXPORT DATA
var bands = [
"B1", "B2", "B3", "B4", "B5", "B6",
"B7", "B8", "B8A", "B9", "B11", "B12"];

var imageConvert = composite.select(bands);
var lonLat = ee.Image.pixelLonLat();
var imageWithCoord = imageConvert.addBands(lonLat);

Export.image.toDrive({
    image: imageWithCoord, 
    description: "Sentinel2_2025_S20_Area", 
    folder: "TA_Export", 
    scale: 20, 
    region: buffer12Mil,
    maxPixels: 1e13, 
    crs: "EPSG:4326", 
    fileFormat: "GeoTIFF"
});