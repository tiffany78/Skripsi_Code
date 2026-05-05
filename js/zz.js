var validMaskAsset = ee.Image(
  'projects/ee-tiffanytasyaagatha/assets/validMask_2025').select('valid_mask').eq(1);
Map.centerObject(validMaskAsset, 8);

Map.addLayer(
  validMaskAsset,
  {palette: ['#00ff00']},
  'validMask',
  true
);
  
// contoh target Batnas
var batnasRaw = ee.Image("projects/ee-tiffanytasyaagatha/assets/Batnas");
var batnasDepth = batnasRaw.multiply(-1).rename('depth');
var batnasMasked = batnasDepth.updateMask(validMaskAsset.eq(1));

Map.addLayer(
  batnasMasked,
  {
    min: 0,
    max: 35,
    palette: ['#08306b', '#2171b5', '#41b6c4', '#a1dab4', '#ffffcc', '#fdae61', '#d73027']
  },
  'BATNAS masked by validMask asset',
  true
);