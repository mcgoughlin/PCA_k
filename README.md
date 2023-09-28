first idea: PCA for finding average kidney shape and normal modes of inter-patient kidney shape variation

Second idea: MLP U-Net for kidney intra-patient shape prediction. This requires paired kidney pointclouds (L and R) for a patients. This requires consistency of the meaning of nodes - i.e., node 1 needs to correspond to the same anatomical point in kidney 1, 5 and 100. However, due to the global attentive properties of linear layers, node 1 on left kidney does *not* need to correspond to the same anatomical location on kidney 2. Thus, node-meaning between kidneys need only be consistent between the populations of left and right kidney.

It would be nice of the MLPUnet was invertible/reversible. I.e, it could be trained on left kidney-to-right kidney prediction, and then be used to predict left kidneys from right kidneys, or visa versa, or a mixture of both simultaneously. To do this, we need an invertible MLPUnet architecture.
