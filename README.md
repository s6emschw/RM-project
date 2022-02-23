# Final Project | Research Module Econometrics & Statistics | Wintersemester 2022 | University of Bonn
by
Carolina Alvarez,
Edoardo Falchi, and
Emily Anne Schwab.

## Abstract

> The prediction accuracy of a traditional least squares model tends to suffer in the presence of multicollinearity and high-dimensionality.  A way of dealing with these issues is the use of regularized regression methods, which sacrifice some bias of the estimated model coefficients in exchange for a sufficient reduction in their variance. Using Monte-Carlo simulations, we explore the statistical properties of three main shrinkage methods - ridge, lasso, and the (naive) elastic net.  Under our data generation set up, we show that the selection of the most suitable method depends on the degrees of dimensionality, sparsity, and multicollinearity that are present in the sample.  We conclude with a real data set application, where we obtain results that are in line with the theoretical discussion of regularized regression and our simulation exercises

## Software implementation

All source code used to generate the results and figures in the paper can be found in the root of the repository.
The calculations and figure generation are all run inside
[Jupyter notebooks](http://jupyter.org/).
The data used in this study is provided in `prostate_dataset.txt` and the sources for the
manuscript text and figures are in `RM-paper`. Slides with main results can be found in `RM-slides`.
