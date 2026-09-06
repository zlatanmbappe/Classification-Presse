#!/usr/bin/env bash
set -e

PACKAGE="classification-presse"
VERSION="1.0.0-1"
ROOT="build/package"
DEB="dist/${PACKAGE}_${VERSION}_all.deb"
SOURCE="dist/${PACKAGE}_${VERSION}_source.tar.gz"

rm -rf build
mkdir -p dist "$ROOT/DEBIAN" "$ROOT/usr/bin" "$ROOT/usr/sbin"
mkdir -p "$ROOT/usr/share/classification-presse"
mkdir -p "$ROOT/usr/share/doc/classification-presse"
mkdir -p "$ROOT/usr/share/applications"
mkdir -p "$ROOT/usr/share/icons/hicolor/scalable/apps"

cp packaging/debian/control "$ROOT/DEBIAN/control"
cp packaging/debian/postinst "$ROOT/DEBIAN/postinst"
cp packaging/debian/postrm "$ROOT/DEBIAN/postrm"
cp packaging/debian/classification-presse "$ROOT/usr/bin/classification-presse"
cp packaging/debian/classification-presse-db-setup "$ROOT/usr/sbin/classification-presse-db-setup"
cp packaging/debian/README.Debian "$ROOT/usr/share/doc/classification-presse/README.Debian"
cp packaging/debian/classification-presse.desktop "$ROOT/usr/share/applications/classification-presse.desktop"
cp packaging/debian/classification-presse.svg "$ROOT/usr/share/icons/hicolor/scalable/apps/classification-presse.svg"
cp LICENSE "$ROOT/usr/share/doc/classification-presse/copyright"

cp -r src config sql docs "$ROOT/usr/share/classification-presse/"
cp requirements.txt README.md LICENSE .env.example "$ROOT/usr/share/classification-presse/"
mkdir -p "$ROOT/usr/share/classification-presse/outputs"

find "$ROOT/usr/share/classification-presse" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$ROOT/usr/share/classification-presse" -type f -name "*.pyc" -delete

find "$ROOT" -type d -exec chmod g-s {} +
find "$ROOT" -type d -exec chmod 755 {} +
chmod 755 "$ROOT/DEBIAN/postinst" "$ROOT/DEBIAN/postrm"
chmod 755 "$ROOT/usr/bin/classification-presse" "$ROOT/usr/sbin/classification-presse-db-setup"

dpkg-deb --root-owner-group --build "$ROOT" "$DEB"

tar --exclude='./dist' --exclude='./build' --exclude='./.venv' --exclude='./.env' \
    --exclude='./outputs/*' --exclude='*/__pycache__' --exclude='*.pyc' \
    -czf "$SOURCE" .

echo
echo "Paquet créé : $DEB"
echo "Sources : $SOURCE"
echo "Installation : sudo apt install ./$DEB"
echo "Lancement : classification-presse"
