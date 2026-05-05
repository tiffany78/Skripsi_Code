Map.centerObject(area2, 10);

// === Parameter Umum ===
var QA_BAND = 'cs_cdf';
var CLEAR_THRESHOLD = 0.55;
var years = [2024];

// === Fungsi Membuat Composite per Tahun ===
function makeComposite(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');
  
  var collection = s2
    .filterDate(start, end)
    .filterBounds(area2)
    .linkCollection(csPlus, [QA_BAND])
    .map(function(img) {
      return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
    });

  // Output composite
  var compositeMedian = collection.median();
  var compositeP10 = collection.reduce(ee.Reducer.percentile([10]));
  var compositeP20 = collection.reduce(ee.Reducer.percentile([20]));

  return {
    median: compositeMedian,
    p10: compositeP10,
    p20: compositeP20
  };
}

// === Fungsi Perhitungan Indeks (NDWI + NDTI) ===
function addIndices(img, prefix) {
  var ndwi = img.normalizedDifference(['B3', 'B8']).rename(prefix + '_NDWI');
  var ndti = img.normalizedDifference(['B4', 'B3']).rename(prefix + '_NDTI');
  
  if(prefix === 'P10'){
    ndwi = img.normalizedDifference(['B3_p10', 'B8_p10']).rename(prefix + '_NDWI');
    ndti = img.normalizedDifference(['B4_p10', 'B3_p10']).rename(prefix + '_NDTI');
  }
  else if (prefix === 'P20'){
    ndwi = img.normalizedDifference(['B3_p20', 'B8_p20']).rename(prefix + '_NDWI');
    ndti = img.normalizedDifference(['B4_p20', 'B3_p20']).rename(prefix + '_NDTI');
  }

  return img.addBands(ndwi).addBands(ndti);
}

function exportTIFF(img, prefix, year) {
  // GEOTIFF
  var bands = [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
    'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'
  ];

  if(prefix === 'P10') {
    bands = [
    'B1_p10', 'B2_p10', 'B3_p10', 'B4_p10', 'B5_p10', 'B6_p10',
    'B7_p10', 'B8_p10', 'B8A_p10', 'B9_p10', 'B11_p10', 'B12_p10'
  ];
  } else if(prefix === 'P20') {
    bands = [
    'B1_p20', 'B2_p20', 'B3_p20', 'B4_p20', 'B5_p20', 'B6_p20',
    'B7_p20', 'B8_p20', 'B8A_p20', 'B9_p20', 'B11_p20', 'B12_p20'
  ];
  }
  var imageConvert = img.select(bands);
  var lonLat = ee.Image.pixelLonLat();
  var imageWithCoord = imageConvert
    .addBands(lonLat);
  Export.image.toDrive({
    image: imageWithCoord,
    description: "Sentinel2_" + prefix + "_" + year,
    folder: 'TA_GEOTIFF',
    scale: 10,
    region: area2,
    maxPixels: 1e13,
    crs: 'EPSG:4326',
    fileFormat: "GeoTIFF"
  });
}

// === Jalankan ===
years.forEach(function(y) {
  var comps = makeComposite(y);

  // Tambahkan indeks untuk setiap composite
  var medianWithIndex = addIndices(comps.median, 'MED');
  var p10WithIndex = addIndices(comps.p10, 'P10');
  var p20WithIndex = addIndices(comps.p20, 'P20');

  // === Visualisasi RGB dasar ===
  Map.addLayer(comps.median.clip(area2), 
               {bands: ['B4','B3','B2'], min:0, max:2500}, 
               'RGB Median ' + y);
  exportTIFF(comps.median, 'MED', y);

  Map.addLayer(comps.p10.clip(area2), 
               {bands: ['B4_p10','B3_p10','B2_p10'], min:0, max:2500}, 
               'RGB P10 ' + y);
  exportTIFF(comps.p10, 'P10', y);

  Map.addLayer(comps.p20.clip(area2), 
               {bands: ['B4_p20','B3_p20','B2_p20'], min:0, max:2500}, 
               'RGB P20 ' + y);
  exportTIFF(comps.p20, 'P20', y);

  // === Visualisasi NDWI ===
  var ndwiVis = {min: -1, max: 1, palette: ['#283618', '#e9edc9', '#8ecae6', '#023047']};
  Map.addLayer(medianWithIndex.select('MED_NDWI').clip(area2),
               ndwiVis, 'NDWI Median ' + y);

  Map.addLayer(p10WithIndex.select('P10_NDWI').clip(area2),
               ndwiVis, 'NDWI P10 ' + y);

  Map.addLayer(p20WithIndex.select('P20_NDWI').clip(area2),
               ndwiVis, 'NDWI P20 ' + y);

  // === Visualisasi NDTI ===
  var ndtiViz = {
    min: -0.5,
    max: 0.5,
    palette: ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
  };
  Map.addLayer(medianWithIndex.select('MED_NDTI').clip(area2),
               ndtiViz, 'NDTI Median ' + y);

  Map.addLayer(p10WithIndex.select('P10_NDTI').clip(area2),
               ndtiViz, 'NDTI P10 ' + y);

  Map.addLayer(p20WithIndex.select('P20_NDTI').clip(area2),
               ndtiViz, 'NDTI P20 ' + y);
});