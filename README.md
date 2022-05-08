# KNN-classifier

### A KNN classifier written from scratch with no dependencies except matplotlib for plotting benchmark results.

#### Benchmarks and examples with various popular datasets can be found inside.


## Benchmark results
| Dataset | Data<br>amount | Observation<br>amount | Test<br>ratio | k range | best<br>k | Best<br>accuracy | Average<br>accuracy | Worst<br>accuracy | Distance<br>metric 
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Iris | 150 | 10 | .2 | `range(3, 20, 2)` | 5 | 100% | 98.3% | 93.3% | Euclidean |
| Diabetes | 768 | 10 | .2 | `range(3, 20, 2)` | 15 | 81% | 74% | 60% | Euclidean |
| Shutter<br><sub>(Scaled down to 1%)</sub> | 434 | 6 | .3 | `range(3, 30, 2)` | 5 | 97.7% | 91.5% | 81% | Euclidean |
| SPECTF heart | 267 | 10 | .2 | `range(3, 20, 2)` | 11 | 87% | 76.4% | 59.3% | Euclidean |
| Abalone<br><sub>(Scaled down to 40%)</sub> | 1670 | 19 | .2 | `range(3, 20, 2)` | 19 | 28.1% | 22.3% | 16.5% | Euclidean |
