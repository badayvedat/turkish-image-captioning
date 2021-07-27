# Change directory to data folder
cd data/

# Clone turkish captions from giddyyupp/turkish-image-captioning
git clone https://github.com/giddyyupp/turkish-image-captioning.git

# Download feature matrix of coco dataset
wget -nc https://cs.stanford.edu/people/karpathy/deepimagesent/coco.zip
unzip coco.zip
cd coco
mkdir feats
mv dataset.json feats/
mv readme.txt feats/
mv vgg_feats.mat feats/

mkdir captions
cp -r ../turkish-image-captioning/MSCOCO/* captions/
cd ..
rm coco.zip

# Download feature matrix of flickr30k
wget -nc https://cs.stanford.edu/people/karpathy/deepimagesent/flickr30k.zip
unzip flickr30k.zip
cd flickr30k
mkdir feats
mv dataset.json feats/
mv readme.txt feats/
mv vgg_feats.mat feats/

mkdir captions
cp -r ../turkish-image-captioning/Flickr30k/train/* captions/
cd ..
rm flickr30k.zip

# Download feature matrix of flickr8k
wget -nc https://cs.stanford.edu/people/karpathy/deepimagesent/flickr8k.zip
unzip flickr8k.zip
cd flickr8k
mkdir feats
mv dataset.json feats/
mv readme.txt feats/
mv vgg_feats.mat feats/

mkdir captions
cd captions
wget https://vision.cs.hacettepe.edu.tr/files/ff1082bf8f613d4a67e4c89a697288e6.zip -O tasviret.zip
unzip tasviret.zip -d tasviret/
rm tasviret.zip
mv tasviret/tasviret8k_captions.json ./
rm -dr tasviret
cd ../../
rm flickr8k.zip

rm -drf turkish-image-captioning
