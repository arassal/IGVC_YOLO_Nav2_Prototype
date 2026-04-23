from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs' / 'ros2_semantic_segmentation_pipeline.png'


def font(size, bold=False):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf' if bold else
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def rounded_box(draw, box, fill, outline, title, body, accent):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=2)
    draw.rectangle((x0, y0, x0 + 10, y1), fill=accent)
    draw.text((x0 + 24, y0 + 18), title, fill=(18, 24, 31), font=font(28, True))
    y = y0 + 58
    for line in body:
        draw.text((x0 + 24, y), line, fill=(45, 55, 66), font=font(19))
        y += 28


def arrow(draw, start, end, color=(56, 71, 85)):
    draw.line((start, end), fill=color, width=4)
    sx, sy = start
    ex, ey = end
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        points = [(ex, ey), (ex - 14 * direction, ey - 8), (ex - 14 * direction, ey + 8)]
    else:
        direction = 1 if ey > sy else -1
        points = [(ex, ey), (ex - 8, ey - 14 * direction), (ex + 8, ey - 14 * direction)]
    draw.polygon(points, fill=color)


def main():
    width, height = 1800, 1180
    img = Image.new('RGB', (width, height), (247, 249, 251))
    draw = ImageDraw.Draw(img)

    draw.text((70, 44), 'ROS 2 Road Segmentation + Cone Detection Pipeline',
              fill=(18, 24, 31), font=font(42, True))
    draw.text((72, 98),
              'Verified local demo path: ROS 2 Jazzy + YOLOPv2 road/lane masks + YOLOv8 traffic cones',
              fill=(76, 88, 100), font=font(24))

    boxes = {
        'input': (70, 180, 470, 380),
        'model': (700, 180, 1100, 380),
        'topics': (1330, 180, 1730, 380),
        'proof': (70, 570, 470, 770),
        'nav': (700, 570, 1100, 770),
        'future': (1330, 570, 1730, 770),
    }

    rounded_box(draw, boxes['input'], (235, 245, 255), (124, 169, 214),
                'Input Frames',
                [
                    'road images with cones',
                    'next: /camera/.../image_raw',
                    'RealSense RGB ready path',
                ], (52, 138, 204))

    rounded_box(draw, boxes['model'], (237, 248, 241), (118, 181, 139),
                'Segmentation Model',
                [
                    'YOLOPv2 TorchScript',
                    'yolopv2.pt, external weight',
                    'drivable area + lane masks',
                ], (47, 145, 86))

    rounded_box(draw, boxes['topics'], (255, 244, 232), (218, 158, 93),
                'Object Model',
                [
                    'Roboflow Logistics YOLOv8',
                    'included 6 MB checkpoint',
                    'traffic cone + people + signs',
                ], (219, 123, 43))

    rounded_box(draw, boxes['proof'], (250, 242, 255), (171, 129, 207),
                'Published Topics',
                [
                    '/seg_ros/lane_mask',
                    '/seg_ros/drivable_mask',
                    '/seg_ros/competition_objects/*',
                    'JSON object detections',
                ], (132, 80, 177))

    rounded_box(draw, boxes['nav'], (236, 249, 249), (99, 178, 184),
                'Verification',
                [
                    'colcon build passes',
                    'combined proof exported',
                    'cones: F1 0.8299',
                    '72 road frames: 0 false cones',
                ], (36, 147, 158))

    rounded_box(draw, boxes['future'], (248, 248, 238), (183, 177, 101),
                'Navigation Next',
                [
                    'live RealSense input',
                    'Nav2 semantic costmap',
                    'cones as obstacle cues',
                ], (155, 148, 50))

    arrow(draw, (470, 280), (700, 280))
    arrow(draw, (1100, 280), (1330, 280))
    arrow(draw, (1530, 380), (1530, 570))
    arrow(draw, (1330, 670), (1100, 670))
    arrow(draw, (700, 670), (470, 670))
    arrow(draw, (270, 570), (270, 380))

    draw.rounded_rectangle((70, 875, 1730, 1065), radius=8,
                           fill=(255, 255, 255), outline=(210, 216, 222), width=2)
    draw.text((100, 905), 'Compatibility Contract', fill=(18, 24, 31), font=font(28, True))
    compatibility = [
        'ROS 2 messages: sensor_msgs/msg/Image, std_msgs/msg/String, vision_msgs/msg/LabelInfo',
        'Verified build command: source /opt/ros/jazzy/setup.bash && colcon build --packages-select seg_ros_bridge',
        'YOLOPv2 segmentation checkpoint stays external; the small YOLOv8 cone/object checkpoint is included.',
    ]
    y = 948
    for line in compatibility:
        draw.text((100, y), line, fill=(45, 55, 66), font=font(21))
        y += 32

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT)
    print(OUT)


if __name__ == '__main__':
    main()
