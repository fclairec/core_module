import re



def helios_waypoint_conversion(cfg):
    print("Converting manual waypoints to CSV format")
    with open(cfg.manual_waypoints_selection) as file:
        waypoints_content = file.readlines()

    pattern = re.compile(r"\(([\d.-]+);([\d.-]+);([\d.-]+)\)")

    tri_lines = [line for line in waypoints_content if "Tri#" in line]
    waypoints = pattern.findall("".join(tri_lines))

    # Convert waypoints to CSV format
    csv_content = "x,y,z\n"
    csv_content += "\n".join([",".join(waypoint) for waypoint in waypoints])

    # Write the CSV content to waypointfile.csv
    with open(cfg.waypoint_file, "w") as file:
        file.write(csv_content)

    print(f"Waypoints written to {cfg.waypoint_file}")
