## Construire le paquet

```bash
chmod +x build_deb.sh
./build_deb.sh
```

Deux fichiers sont créés dans `dist/` :

- `classification-presse_1.0.0-1_all.deb`
- `classification-presse_1.0.0-1_source.tar.gz`

## Installer

```bash
sudo apt install ./dist/classification-presse_1.0.0-1_all.deb
```

## Lancer

```bash
classification-presse
```

