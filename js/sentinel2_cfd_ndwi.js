Map.centerObject(area2, 10);

// ==== EXPORT METODE 2 ====
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

  var compositeMedian = collection.median();
  return { median: compositeMedian };
}

// === Fungsi Perhitungan NDWI ===
function addNDWI(img, prefix) {
  var ndwi = img.normalizedDifference(['B3', 'B8']).rename(prefix + '_NDWI');
  return img.addBands(ndwi);
}

// === Fungsi Export ===
function exportTIFF(img, prefix, year) {

  var bands = [
    // Sentinel-2 band asli
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
    'B7', 'B8', 'B8A', 'B9','B11', 'B12',

    // NDWI dan NDWIclass
    prefix + '_NDWI',
    prefix + '_NDWIclass'
  ];

  var imageConvert = img.select(bands);
  var lonLat = ee.Image.pixelLonLat();
  var imageWithCoord = imageConvert.addBands(lonLat);

  Export.image.toDrive({
    image: imageWithCoord,
    description: "Sentinel2_NDWI_20_" + prefix + "_" + year,
    folder: 'TA_GEOTIFF',
    scale: 20,
    region: area2,
    maxPixels: 1e13,
    crs: 'EPSG:4326',
    fileFormat: "GeoTIFF"
  });
}

// === Jalankan ===
years.forEach(function(y) {

  var comps = makeComposite(y);

  // Tambahkan NDWI
  var medianWithNDWI = addNDWI(comps.median, 'MED');

  // Ambil NDWI
  var ndwi = medianWithNDWI.select('MED_NDWI');

  // Hitung NDWI Class
  var ndwiClass = ndwi.expression(
    "(ndwi <= 0) ? 1" +        // Daratan
    ": (ndwi <= 0.2) ? 2" +    // Tanah lembab
    ": (ndwi <= 0.5) ? 3" +    // Perairan dangkal
    ": 4",                     // Perairan dalam
    { ndwi: ndwi }
  ).rename('MED_NDWIclass');

  // Tambahkan NDWIclass sebagai band
  medianWithNDWI = medianWithNDWI.addBands(ndwiClass);

  // Visualisasi RGB
  Map.addLayer(
    comps.median.clip(area2), 
    {bands: ['B4','B3','B2'], min:0, max:2500}, 
    'RGB Median ' + y
  );

  // Export (hanya band S2 + NDWI + NDWIclass)
  exportTIFF(medianWithNDWI, 'MED', y);

});
