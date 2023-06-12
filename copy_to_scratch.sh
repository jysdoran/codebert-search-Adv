mkdir -p $SCRATCHBIG

rm -rf $SCRATCHBIG/dataset/
rm -rf $SCRATCHBIG/microsoft/

cp -r microsoft/ $SCRATCHBIG
cp -r dataset/ $SCRATCHBIG
