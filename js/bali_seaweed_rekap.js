Map.centerObject(bali_seaweed, 10);

// === Parameter Umum ===
var QA_BAND = 'cs_cdf';
var CLEAR_THRESHOLD = 0.55;
var years = [2024];

// === Fungsi Membuat Composite per Tahun ===
function makeComposite(year) {
  var start = ee.Date.fromYMD(year, 1, 1);
  var end = start.advance(1, 'year');
  
  var composite = s2
    .filterDate(start, end)
    .filterBounds(bali_seaweed)
    .linkCollection(csPlus, [QA_BAND])
    .map(function(img) {
      return img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD));
    })
    .median();
  // Visualisasi RGB
  var visParams = {bands: ['B4', 'B3', 'B2'], min: 0, max: 2500};
  // Tambahkan ke Map
  Map.addLayer(composite.clip(bali_seaweed), visParams, 'RGB ' + year);
  
  // GEOTIFF
  var bands = [
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6',
    'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'
  ];
  var imageConvert = composite.select(bands);
  var lonLat = ee.Image.pixelLonLat();
  var imageWithCoord = imageConvert
    .addBands(lonLat);
  var sampled = imageWithCoord.sample({
    region: bali_seaweed,
    scale: 10,
    geometries: true
  });
  Export.image.toDrive({
    image: imageConvert,
    description: "Sentinel2_" + year,
    folder: 'TA_GEOTIFF',
    scale: 10,
    region: bali_seaweed,
    crs: 'EPSG:4326',
    fileFormat: "GeoTIFF"
  });
  
  return composite;
}

// === Hitung koefisien atenuasi antar band (k) ===
function computeK(image, band1, band2, waterMask) {
  var log1 = image.select(band1).log();
  var log2 = image.select(band2).log();

  var stats1 = log1.updateMask(waterMask).reduceRegion({
    reducer: ee.Reducer.stdDev(),
    geometry: bali_seaweed,
    scale: 10,
    maxPixels: 1e10
  });
  var stats2 = log2.updateMask(waterMask).reduceRegion({
    reducer: ee.Reducer.stdDev(),
    geometry: bali_seaweed,
    scale: 10,
    maxPixels: 1e10
  });

  var std1 = ee.Number(stats1.get(band1));
  var std2 = ee.Number(stats2.get(band2));

  return std1.divide(std2);
}

// === Fungsi Depth Invariant Index (Lyzenga) ===
function computeDII(image, band1, band2, k) {
  var log1 = image.select(band1).log();
  var log2 = image.select(band2).log();
  var dii = log1.subtract(log2.multiply(k)).rename('DII_' + band1 + '_' + band2);
  return dii;
}

// === Fungsi Perhitungan Indeks ===
function addIndices(image) {

  // --- Mask untuk area air dari NDWI ---
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var waterMask = ndwi.gt(0);   // NDWI > 0 diasumsikan air
  
  // --- Hitung koefisien k untuk pair band ---
  var k_B2_B3 = computeK(image, 'B2', 'B3', waterMask);
  var k_B2_B4 = computeK(image, 'B2', 'B4', waterMask);
  var k_B3_B4 = computeK(image, 'B3', 'B4', waterMask);

  // --- Hitung DII ---
  var DII_B2_B3 = computeDII(image, 'B2', 'B3', k_B2_B3);
  var DII_B2_B4 = computeDII(image, 'B2', 'B4', k_B2_B4);
  var DII_B3_B4 = computeDII(image, 'B3', 'B4', k_B3_B4);

  // --- Tambahkan semua indeks lainnya ---
  var bsi = image.expression(
    '((SWIR + RED) - (NIR + BLUE)) / ((SWIR + RED) + (NIR + BLUE))',
    {
      'SWIR': image.select('B11'),
      'RED': image.select('B4'),
      'NIR': image.select('B8'),
      'BLUE': image.select('B2')
    }).rename('BSI');
  
  var lyzengaRatio = image.select('B2').divide(image.select('B3')).rename('LYZ_RATIO');
  var ndti = image.normalizedDifference(['B4','B3']).rename('NDTI');
  var si = image.select('B4').divide(image.select('B3')).rename('SI');

  return image
    .addBands(ndwi)
    .addBands(bsi)
    .addBands(lyzengaRatio)
    .addBands(ndti)
    .addBands(si)
    .addBands(DII_B2_B3)
    .addBands(DII_B2_B4)
    .addBands(DII_B3_B4);
}

// === Jalankan Loop ===
years.forEach(function(y) {
  var composite = makeComposite(y);
  // Konversi ke ImageCollection
  var compositesCol = ee.ImageCollection(composite);
  // Tambahkan indeks ke setiap composite
  var compositesWithIndex = compositesCol.map(addIndices);
  // === Cek hasil ===
  var sampleImage = ee.Image(compositesWithIndex.first()).clip(bali_seaweed);
  var vssi = sampleImage.select('VSSI');
  var classified = vssi
  .where(vssi.lte(-1.64), 1)
  .where(vssi.gt(-1.64).and(vssi.lte(-1.33)), 2)
  .where(vssi.gt(-1.33).and(vssi.lte(-1.03)), 3)
  .where(vssi.gt(-1.03).and(vssi.lte(-0.729)), 4)
  .where(vssi.gt(-0.729).and(vssi.lte(-0.427)), 5);

  var ndtiViz = {
    min: -0.5,
    max: 0.5,
    palette: ['blue', 'cyan', 'green', 'yellow', 'orange', 'red']
  };

  var sedViz = {
    min: 0.5,
    max: 2,
    palette: ['blue', 'green', 'yellow', 'orange', 'red']
  };

  Map.addLayer(sampleImage.select('NDWI'), {min: -1, max: 1, palette: ['#283618', '#e9edc9', '#8ecae6', '#023047']}, 'NDWI ' + y );
  Map.addLayer(sampleImage.select('BSI'), {min: -1, max: 1, palette: ['blue', 'white', 'red']}, 'BSI');
  Map.addLayer(sampleImage.select('LYZ'), {min: 0.5, max: 2, palette: ['darkblue', 'cyan', 'yellow']}, 'Lyzenga Ratio');
  Map.addLayer(classified, {min: 1, max: 5, palette: ['#ff0000', '#ffa500', '#ffff00', '#00ff00', '#0000ff']}, 'Salinity');
  Map.addLayer(sampleImage.select('NDTI'), ndtiViz, 'NDTI');
  Map.addLayer(sampleImage.select('DII_B2_B3'), {min:-0.5, max:0.5, palette:['blue','cyan','yellow','red']},'DII B2-B3');
  Map.addLayer(sampleImage.select('DII_B3_B4'), {min:-0.5, max:0.5, palette:['blue','green','yellow','red']},'DII B3-B4');
  Map.addLayer(sampleImage.select('DII_B2_B4'), {min:-0.5, max:0.5, palette:['blue','green','yellow','red']},'DII B2-B4');
});

// === LEGEND PANEL SETUP ===
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px',
    backgroundColor: 'rgba(255, 255, 255, 0.8)',
    width: '350px'
  }
});

// === JUDUL LEGEND ===
var legendTitle = ui.Label({
  value: 'Legenda Indeks',
  style: {
    fontWeight: 'bold',
    fontSize: '14px',
    margin: '0 0 6px 0',
    textAlign: 'center'
  }
});
legend.add(legendTitle);

// === FUNGSI PEMBUAT ROW LEGEND ===
function makeRow(color, name) {
  var colorBox = ui.Label({
    style: {
      backgroundColor: color,
      padding: '8px',
      margin: '0 6px 4px 0',
      border: '1px solid black'
    }
  });
  var description = ui.Label({
    value: name,
    style: { margin: '0 0 4px 0', fontSize: '12px' }
  });

  return ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
}

// === PILIH INDEKS UNTUK LEGEND ===
var selectIndex = ui.Select({
  items: ['NDWI', 'BSI', 'LYZ', 'VSSI', 'NDTI', 'SI'],
  value: 'NDWI',
  style: { width: '100%', margin: '4px 0' },
  onChange: function(index) {
    legend.clear();
    legend.add(legendTitle);
    legend.add(selectIndex);

    if (index === 'NDWI') {
      legend.add(makeRow('#283618', 'NDWI -1.0 to -0.5 → Daratan kering / non-air'));
      legend.add(makeRow('#e9edc9', 'NDWI -0.5 to 0.0 → Tanah lembab / area transisi'));
      legend.add(makeRow('#8ecae6', 'NDWI 0.0 to 0.5 → Perairan dangkal / vegetasi air'));
      legend.add(makeRow('#023047', 'NDWI 0.5 to 1.0 → Perairan dalam / jernih'));
    } else if (index === 'BSI') {
      legend.add(makeRow('blue', '< 0 : Air / vegetasi'));
      legend.add(makeRow('white', '≈ 0 : Transisi'));
      legend.add(makeRow('red', '> 0 : Pasir / darat'));
    } else if (index === 'LYZ') {
      legend.add(makeRow('darkblue', '0.5–0.8 : Air dalam'));
      legend.add(makeRow('cyan', '0.8–1.2 : Air dangkal'));
      legend.add(makeRow('yellow', '> 1.2 : Karang'));
    } else if (index === 'VSSI') {
      legend.add(makeRow('#ff0000', 'merah'));
      legend.add(makeRow('#ffa500', 'orange'));
      legend.add(makeRow('#ffff00', 'kuning'));
      legend.add(makeRow('#00ff00', 'hijau'));
      legend.add(makeRow('#0000ff', 'biru'));
    } else if (index === 'NDTI') {
      legend.add(makeRow('blue', 'keruh'));
      legend.add(makeRow('cyan', ' '));
      legend.add(makeRow('green', ' '));
      legend.add(makeRow('yellow', ' '));
      legend.add(makeRow('orange', ' '));
      legend.add(makeRow('red', 'jernih'));
    } else if (index === 'SI') {
      legend.add(makeRow('blue', 'keruh'));
      legend.add(makeRow('green', ' '));
      legend.add(makeRow('yellow', ' '));
      legend.add(makeRow('orange', ' '));
      legend.add(makeRow('red', 'jernih'));
    }
  }
});
legend.add(selectIndex);
Map.add(legend);