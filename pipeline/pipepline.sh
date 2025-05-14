# bash
cd ../torchserve/
sudo apptainer run my_animated.sif -p 8080:8080 -p 8081:8081
cd ../examples/
sleep 20
python image_to_annotations.py ../pipeline/inputs/robot.jpg  characters/robot
sudo /home/martin-barry/Desktop/HES-SO/AnimatedDrawings/.venv/bin/torchserve --stop

cd ../
python live_demo.py

cd pipeline/
# python fix_annotations.py /home/martin-barry/Desktop/HES-SO/AnimatedDrawings/examples/characters/robot/