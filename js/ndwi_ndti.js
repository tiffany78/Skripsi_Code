Map.centerObject(bali_seaweed, 15.5);
// Map.centerObject(asmat, 8);

// ASMAT 
// Pemanggilan Asset
var batas = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Batas2024");
var asmat = batas.filter(
  ee.Filter.stringContains('WADMKK', 'Asmat')
);
// Pembuatan Layer
Map.addLayer(asmat, {color: 'blue', strokeWidth : 10},'batas24');
Map.centerObject(asmat, 8);

// PEMBUATAN AOI
var garisPantai = ee.Geometry.LineString([
  batasAtas.coordinates(),
  batasBawah.coordinates()
]);
var jarak12Mil = 12 * 1852; // meter
var zonaPesisir = garisPantai.buffer(jarak12Mil);

// ROI 12 MIL
var garisPantaiAsmat = batas50K.filterBounds(asmat);
var coastlineAll = garisPantaiAsmat.geometry().dissolve();
var coastline = coastlineAll.intersection(
  asmat.geometry().buffer(100),
  ee.ErrorMargin(1)
);
Map.addLayer(coastline, {color:'red'}, 'Coastline Asmat FIX');

var panjang = coastline.length();
var interval = 1000;
var distances = ee.List.sequence(0, panjang, interval);
var segments = coastline.cutLines(distances);
var titikSampling = ee.FeatureCollection(
  segments.geometries().map(function(g) {
    return ee.Feature(ee.Geometry(g).centroid());
  })
);
Map.addLayer(titikSampling, {color:'red'}, 'Titik 1km');

var jarak12Mil = 12 * 1852; // meter
var buffer12Mil = titikSampling.map(function(f) {
  return f.buffer(jarak12Mil);
});
Map.addLayer(buffer12Mil, {color:'green'}, 'Buffer 12 Mil');

// VISUALUSASI CITRA
var csPlus=ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED");
var QA_BAND = "cs_cdf";
var CLEAR_THRESHOLD = 0.55;

var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED");
var composite = s2
    .filterDate("2025-01-01", "2025-12-31")
    .linkCollection(csPlus, [QA_BAND])
    .map(function(img) {
      return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));})
    ;

var baliImg = composite.filterBounds(bali2).median();
var asmatImg = composite.filterBounds(buffer12Mil).median();

// HITUNG NDWI
var baliNDWI = baliImg.normalizedDifference(['B3','B8']).rename('NDWI');
var asmatNDWI = asmatImg.normalizedDifference(['B3','B8']).rename('NDWI');
// NDTI = (Red - Green)/(Red + Green)
var baliNDTI = baliImg.normalizedDifference(['B4','B3']).rename('NDTI');
var asmatNDTI = asmatImg.normalizedDifference(['B4','B3']).rename('NDTI');

// MASK AIR 
var baliWater = baliNDWI.gt(0);
var asmatWater = asmatNDWI.gt(0);

baliNDTI = baliNDTI.updateMask(baliWater);
asmatNDTI = asmatNDTI.updateMask(asmatWater);

// HITUNG STATISTIK
function getStats(image, region, name){
  var stats = image.reduceRegion({
    reducer: ee.Reducer.mean()
      .combine(ee.Reducer.stdDev(), '', true),
    geometry: region,
    scale: 10,
    maxPixels: 1e13
  });
  print(name, stats);
}

// NDWI stats
getStats(baliNDWI, bali, 'BALI NDWI');
getStats(asmatNDWI, asmat, 'ASMAT NDWI');

// NDTI stats
getStats(baliNDTI, bali, 'BALI NDTI');
getStats(asmatNDTI, asmat, 'ASMAT NDTI');


// HISTOGRAM PERBANDINGAN
var chartNDWI = ui.Chart.image.histogram({
  image: baliNDWI.addBands(asmatNDWI),
  region: bali.merge(asmat),
  scale: 10,
  maxPixels: 1e13
}).setOptions({
  title: 'Histogram NDWI Bali vs Asmat',
  hAxis: {title: 'NDWI'},
  vAxis: {title: 'Frequency'}
});
print(chartNDWI);

var chartNDTI = ui.Chart.image.histogram({
  image: baliNDTI.addBands(asmatNDTI),
  region: bali.merge(asmat),
  scale: 10,
  maxPixels: 1e13
}).setOptions({
  title: 'Histogram NDTI Bali vs Asmat',
  hAxis: {title: 'NDTI'},
  vAxis: {title: 'Frequency'}
});
print(chartNDTI);

// VISUALISASI
Map.addLayer(baliNDTI, {min:-0.2, max:0.5, palette:['blue','yellow','red']}, 'Bali NDTI');
Map.addLayer(asmatNDTI, {min:-0.2, max:0.5, palette:['blue','yellow','red']}, 'Asmat NDTI');

// Pemanggilan Asset
var batas50K = ee.FeatureCollection("projects/ee-tiffanytasyaagatha/assets/Garis50K");
// Pembuatan Layer
Map.addLayer(batas50K, {color: 'yellow', strokeWidth : 10},'batas50K');