

// Parameters
base_size = [35 + PI + 15, 75, PI]; // width, depth, height
pillar_size = [PI, 37, 29.6 - PI]; // width, depth, height
pillar_spacing = 33; // distance between the centers of the pillars
hole_diameter = 3; // Diameter of the screw holes
top_edge_offset = 2.5; // Offset from the top edge for hole placement (prev:5)
side_edge_offset = 10; // Offset from the side edge for hole placement (prev:11)
hole_facets = 50; // Number of facets for the hole cylinder


// Modules
module base() {
    cube(base_size);
}

module pillar_with_holes() {
    difference() {
        cube(pillar_size);
        // Create horizontal holes that go all the way through the pillar
        for (y = [side_edge_offset, pillar_size[1] - side_edge_offset]) {
            translate([0, y, pillar_size[2] - top_edge_offset]) {
                rotate([0, 90, 0]) cylinder(h = pillar_size[0], d = hole_diameter, $fn = hole_facets);
            }
        }
    }
}

// Create the base
translate([-base_size[0]/2 - 15/2, -base_size[1]/2, 0]) {
    base();
}

// Create the first pillar with holes
translate([-pillar_spacing/2 - pillar_size[0]/2, -pillar_size[1]/2, base_size[2]]) {
    pillar_with_holes();
}

// Create the second pillar with holes
translate([pillar_spacing/2 - pillar_size[0]/2, -pillar_size[1]/2, base_size[2]]) {
    pillar_with_holes();
}

