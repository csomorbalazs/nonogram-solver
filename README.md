# Solving nonograms using neural networks

Try our neural network based nonogram solver at [nonogram-solver.web.app](https://nonogram-solver.web.app/)

![Picture of a 6x6 nonogram solved by our algorithm](nonogram.png)

## Commands  
Usage of the scripts in src/

### Generating data 
```
datagen.py

-s, --samples Number of generated samples

-r, --rows Number of rows in a nonogram

-c, --columns Number of columns in a nonogram

-t, --train Train split [0.0, 1.0]

-v, --valid Valid split [0.0, 1.0]

-o, --output Output folder
```

### Creating a model 
```
models.py

-i, --input Input folder, consisting of training and validation data, default value: ./

-o, --output Compiled model is saved to here, default value: compiled/
```

### Training the model  

```
train.py

-i, --input   Input folder, consisting training and validation data , default value: ./

-l, --logdir   TensorBoard logging directory, default value: logs/scalars

-e, --earlystopping   Early stopping patience, default value: 10

-m, --model Model path, default value:  models/

-c, --checkpoints  Use checkpoints (save best model during training), default value: False

-o, --output Trained model is saved to here, default value: trained
```

### Evaluating the model  

```
evaluate.py

-i, --input  Input folder, consisting test data, default value: ./

-m, --model  Model path, default value:  trained/
```
