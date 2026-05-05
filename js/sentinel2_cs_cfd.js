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

  // Output composite
  var compositeMedian = collection.median();

  return {
    median: compositeMedian
  };
}

// === Fungsi Perhitungan Indeks (NDWI + NDTI) ===
function addIndices(img, prefix) {
  var ndwi = img.normalizedDifference(['B3', 'B8']).rename(prefix + '_NDWI');
  var ndti = img.normalizedDifference(['B4', 'B3']).rename(prefix + '_NDTI');
  return img.addBands(ndwi).addBands(ndti);
}

function exportTIFF(img, prefix, year) {
  // GEOTIFF
  var bands = [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
    'B7', 'B8', 'B8A', 'B9','B11', 'B12'
  ];

  var imageConvert = img.select(bands);
  var lonLat = ee.Image.pixelLonLat();
  var imageWithCoord = imageConvert
    .addBands(lonLat);
  Export.image.toDrive({
    image: imageWithCoord,
    description: "Sentinel2_" + prefix + "_" + year,
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

  // Tambahkan indeks untuk setiap composite
  var medianWithIndex = addIndices(comps.median, 'MED');

  // === Visualisasi RGB dasar ===
  Map.addLayer(
    comps.median.clip(area2), 
    {bands: ['B4','B3','B2'], min:0, max:2500}, 
    'RGB Median ' + y
  );
  exportTIFF(comps.median, 'MED', y);

  // === Visualisasi NDWI ===
  // Ambil NDWI median
  var ndwi = medianWithIndex.select('MED_NDWI');
  
  // Klasifikasi NDWI menjadi 4 kelas
  var ndwiClass = ndwi.expression(
    "(ndwi <= 0) ? 1" +        // 1 = Daratan
    ": (ndwi <= 0.2) ? 2" +    // 2 = Tanah lembab
    ": (ndwi <= 0.5) ? 3" +    // 3 = Perairan dangkal
    ": 4",                     // 4 = Perairan dalam
    {
      ndwi: ndwi
    }
  );
  
  var ndwiClassVis = {
    min: 1,
    max: 4,
    palette: [
      '#283618', // 1 = Daratan
      '#ddb892', // 2 = Tanah lembab
      '#8ecae6', // 3 = Perairan dangkal
      '#023047'  // 4 = Perairan dalam
    ]
  };
  Map.addLayer(ndwiClass.clip(area2), ndwiClassVis, 'NDWI Class ' + y);

  // ============================================================
  // === Visualisasi NDTI HANYA untuk area air (NDWI > 0) ===
  // ============================================================

  // Masker area air
  var waterMask = ndwi.gt(0);

  // Mask NDTI dengan waterMask
  var ndtiMasked = medianWithIndex.select('MED_NDTI')
                                  .updateMask(waterMask);

  // Style warna untuk NDTI air
  var ndtiViz = {
    min: -0.5,
    max: 0.5,
    palette: ['#023e8a', '#48cae4', '#caf0f8', '#deab90', '#a98467', '#432818']
  };

  Map.addLayer(
    ndtiMasked.clip(area2),
    ndtiViz,
    'NDTI (Only Water) ' + y
  );
});