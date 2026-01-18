# Tea Base Mapping (Draft)

This mapping powers the tea base mix chart and helps audit how bases are assigned.

## Base Rules (Priority Order)

1. Matcha: Any item/modifier/category containing `matcha`.
2. Genmai Green: Any item/modifier/category containing `genmai` (with or without `green`).
3. TGY Oolong: Any item/modifier/category containing `tgy` or `oolong` (includes osmanthus honey).
4. Buckwheat Barley: Any item/modifier/category containing `buckwheat` or `barley`.
5. Black: Any item/modifier/category containing `black tea` or `black`.
6. Green: Any item/modifier/category containing `green`.
7. Four Seasons: Any item/modifier/category containing `four seasons` (only applies when Green is not present).
8. Unknown: When no base keywords are found.

Notes:
- Green is applied before Four Seasons so fruit tea choices stay accurate.
- Matcha always wins, even in blends.

## Signature Overrides

These override the base rules when present in the item name:

- Taiwanese Retro -> Black
- Pistachio Mist -> Genmai Green
- Brown Sugar Mist -> TGY Oolong
- Grapefruit Bloom -> Four Seasons

## Fruit Tea Choice (Modifiers Applied)

Examples from the data:
- `Fresh Fruit Tea ...` + `Green Tea, ...` -> Green
- `Fresh Fruit Tea ...` + `Four Seasons Tea, ...` -> Four Seasons

If you add or rename menu items, update these rules to keep the mapping accurate.
