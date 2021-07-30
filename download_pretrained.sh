# Since It's my drive account I do not guarantee that this link will work in the future.
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WftjS5CvGItebD9Vav7oHG7I1eBIpmLx' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WftjS5CvGItebD9Vav7oHG7I1eBIpmLx" -O 12heads.zip && rm -rf /tmp/cookies.txt

unzip -q 12heads.zip -d 12heads
mv 12heads/checkpoints/* checkpoints/
mv *.pkl ./
rm -dr 12heads
