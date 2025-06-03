# bash
cd ../torchserve/
sudo apptainer run my_animated.sif -p 8080:8080 -p 8081:8081
cd ../scripts/
sleep 20
python image_to_annotations.py ../drawings/input_images/$1.jpg  ../drawings/characters/$1
sudo /home/martin-barry/Desktop/HES-SO/AnimatedDrawings/.venv/bin/torchserve --stop

python live_demo.py

