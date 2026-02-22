# Dataset format

Place your images under class folders:

```text
dataset/
├── acne/
├── eczema/
├── melanoma/
└── psoriasis/
```

Accepted formats: jpg, jpeg, png, bmp, webp.

Invalid/corrupted images are automatically removed by `clean_dataset()` during training.
