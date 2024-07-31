input = n_kv superformula, parameters for super formulars

output = control points (2*N), weights, knot vectors

generator: generating training data: get control points from the super fomulars, randomly generating weights and knot vectors

distinguisher: ground truth - the super formula

calculate loss 

I have a superfomula that can generate shapes based on the parameter input. 
I want to describe the shape in nurbs representation in control points, weights, and knot vectors. 
I want to build a generative model that takes the parameters for the super formula and number of knot vectors and generate the correspoinding contorl points, weights and number of knot vectors that's closest to the superformula
how should I go about building and training this model? I'm thinking to use GAN model structure but let me know which model structure will be better



