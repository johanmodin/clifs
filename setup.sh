mkdir -p data/output
mkdir -p data/input
while true; do
    read -p "Do you wish to download the default traffic video? (y/n): " yn
    case $yn in
        [Yy]* ) curl -o data/input/sherbrooke_video.avi https://www.jpjodoin.com/urbantracker/dataset/sherbrooke/sherbrooke_video.avi; break;;
        [Nn]* ) echo "Not downloading video dataset"; exit;;
        * ) echo "Please answer yes or no.";;
    esac
done
