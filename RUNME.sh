make -j8

#train
bin/cytonRl --env roms/breakout.bin --mode train --saveModel model/model --showScreen 1

#fast train
# This command turns off the game window which makes the program run much faster. The game window can be brought back by creating a empty file "mode/model.screen", or turned off again by deleting that file.
# bin/cytonRl --env roms/breakout.bin --mode train --saveModel model/model --showScreen 0


#test
#bin/cytonRl --mode test --env roms/breakout.bin  --loadModel model/model.100000000 --showScreen 1

#test using our trained model
bin/cytonRl --mode test --env roms/breakout.bin  --loadModel model-trained/model --showScreen 1

#fast test
#bin/cytonRl --mode test --env roms/breakout.bin  --loadModel model-trained/model --showScreen 0

