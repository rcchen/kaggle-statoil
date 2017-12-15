# statoil-iceberg-classifier

In which we try to distinguish icebergs from ships.

[Kaggle](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge)

## Notes

Will be updated as a running log of things that have been tried.

Data is provided in the following format:

```ts
{
  id: string,
  band1: number[],
  band2: number[],
  inc_angle: number | "na",
  is_iceberg: boolean
}
```

where any `number` can be a `float`.

### Linear SVM

First attempt is just to use linear SVM on something super simple like the
values of the first band. It turns out to be somewhat of a disaster

```
             precision    recall  f1-score   support

          0       0.50      1.00      0.67        81
          1       0.00      0.00      0.00        80

avg / total       0.25      0.50      0.34       161

[[81  0]
 [80  0]]
```

It assigns everything to one category. Clear that SVM is not a good approach
to use here.

### KMeans

```
             precision    recall  f1-score   support

          0       0.55      0.42      0.48        81
          1       0.53      0.65      0.58        80

avg / total       0.54      0.53      0.53       161

[[34 47]
 [28 52]]
```

Slightly "better" approach in the sense that it's not just blindly outputting
the same value over and over again.

### Neural nets

First idea is to just throw everything 

## References

Will be updated as more research is done.

* Combining polarimetric channels for better ship detection results
  * https://earth.esa.int/c/document_library/get_file?folderId=409229&name=DLFE-5566.pdf
  * HH works better at high incidence angles (band_1) so may need to do something there
* Ship-Iceberg Discrimination with Convolutional Neural Networks in High Resolution SAR Images
  * http://elib.dlr.de/99079/2/2016_BENTES_Frost_Velotto_Tings_EUSAR_FP.pdf

