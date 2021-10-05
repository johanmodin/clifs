mkdir -p data/output
mkdir -p data/input
while true; do
    read -p "Do you wish to download the default traffic video? (y/n): " yn
    case $yn in
[Yy]* ) curl -O data/input https://www.jpjodoin.com/urbantracker/dataset/sherbrooke/sherbrooke_video.avi -o data/input/sherbrooke_video.avi; break;;
[Nn]* ) echo "Not downloading video dataset"; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
