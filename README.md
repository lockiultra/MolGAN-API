# MolGAN-API
API implementation of https://github.com/lockiultra/SCAMT

Docker image - https://hub.docker.com/repository/docker/lockiultra/molgan_api/general

Маршрутизация запросов к API:
```
GET /molgan/sample_mol - возвращает сгенерированный SMILES молекулы
Работат на обученной модели вариационного автоэнкодера JTVAE.
```

```
GET /molgan/predict?smiles={{ smiles }} - возвращает json в формате {'disease_1': 'probability_1', ...}
На текущий момент предсказания осуществляются для 8 категорий болезней на основе MPNN моделей.
```
