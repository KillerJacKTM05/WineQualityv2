   fixed acidity  volatile acidity  citric acid  ...  sulphates  alcohol  quality
0            7.4              0.70         0.00  ...       0.56      9.4        5
1            7.8              0.88         0.00  ...       0.68      9.8        5
2            7.8              0.76         0.04  ...       0.65      9.8        5
3           11.2              0.28         0.56  ...       0.58      9.8        6
4            7.4              0.70         0.00  ...       0.56      9.4        5

Quality score range: 3 to 8
quality bin
medium    1319
high       217
low         63

                 Feature     F-Score       p-Value
10               alcohol  158.777591  1.291616e-63
1       volatile acidity  102.403966  1.436358e-42
2            citric acid   44.840551  1.130994e-19
9              sulphates   36.519975  3.103071e-16
6   total sulfur dioxide   21.956997  3.917674e-10
7                density   18.765167  8.805122e-09
0          fixed acidity   13.168538  2.126430e-06
8                     pH    9.410306  8.650976e-05
5    free sulfur dioxide    9.300891  9.639020e-05
4              chlorides    8.259666  2.699663e-04
3         residual sugar    2.323580  9.825354e-02

   alcohol  volatile acidity  ...  total sulfur dioxide  quality bin
0      9.4              0.70  ...                  34.0       medium
1      9.8              0.88  ...                  67.0       medium
2      9.8              0.76  ...                  54.0       medium
3      9.8              0.28  ...                  60.0       medium
4      9.4              0.70  ...                  34.0       medium

          Feature 1             Feature 2  Distance Consistency
0           alcohol      volatile acidity              0.835522
7       citric acid             sulphates              0.834271
1           alcohol           citric acid              0.833021
2           alcohol             sulphates              0.832395
3           alcohol  total sulfur dioxide              0.826767
9         sulphates  total sulfur dioxide              0.819887
8       citric acid  total sulfur dioxide              0.819262
6  volatile acidity  total sulfur dioxide              0.818637
5  volatile acidity             sulphates              0.812383
4  volatile acidity           citric acid              0.805503

--- Linear Regression Results using ANOVA ---
Mean Squared Error (MSE): 0.415
R-squared (R^2) Score: 0.357

--- RandomForest Regression Results using ANOVA ---
MSE: 0.3382915625
R2: 0.47573920939524195

--- Linear Regression Results without using ANOVA ---
Mean Squared Error (MSE): 0.406
R-squared (R^2) Score: 0.370

--- RandomForest Regression Results without using ANOVA ---
MSE: 0.3220990625
R2: 0.5008332223500213

--- RandomForest Classification Results ---
              precision    recall  f1-score   support

        high       0.55      0.72      0.63        43
         low       0.20      0.15      0.17        13
      medium       0.91      0.88      0.89       264

    accuracy                           0.82       320
   macro avg       0.55      0.58      0.56       320
weighted avg       0.83      0.82      0.83       320

[[ 31   0  12]
 [  0   2  11]
 [ 25   8 231]]
