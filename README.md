# Cat and Mouse
Game made in pygame and is training using the q learning method. Made on a 10x10 grid with all objects statically placed on the board.
We can see the progression as the episodes go on. A couple issues I have with this is that every time the model is trained it has to initialize pygame
and run it through that way which takes significantly longer time than just training based on a np.matrix. Other thing is if there's a better way of 
adjusting the learning and discount values as to be able to change them a couple episodes must first pass which takes quite a while.

## Untrained on episode 1-3

https://user-images.githubusercontent.com/83138403/223722220-11afdee2-d594-484a-a883-f0869f639c46.mov

## Trained after 5000 episodes

https://user-images.githubusercontent.com/83138403/223722229-767f004d-3489-4eb3-89bf-f0f67fceb642.mov

