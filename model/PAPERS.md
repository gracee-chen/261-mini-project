# Model References

| Model | Paper | Val Acc |
|-------|-------|---------|
| ResNet-50 | He et al., "Deep Residual Learning for Image Recognition", CVPR 2016 | 97.36% |
| EfficientNet-B2 | Tan & Le, "EfficientNet: Rethinking Model Scaling for CNNs", ICML 2019 | 96.98% |
| ViT-B/16 | Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", ICLR 2021 | 95.70% |
| ConvNeXt-Tiny | Liu et al., "A ConvNet for the 2020s", CVPR 2022 | **97.66%** |
| SVM + ResNet-18 | Classical ML: frozen ResNet-18 feature extractor + RBF SVM | 89.89% (val) / 90.45% (test) |

## BibTeX Keys

- `he2016deep` - ResNet
- `tan2019efficientnet` - EfficientNet
- `dosovitskiy2020image` - Vision Transformer
- `liu2022convnet` - ConvNeXt
- `donahue2014decaf` - DeCAF (CNN features for SVM)
- `fei2004learning` - Caltech-101
- `krizhevsky2012imagenet` - AlexNet
