struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) vert_pos: vec3<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Define vertices for the first triangle
    if (in_vertex_index < 3u) {
        let x = f32(1 - i32(in_vertex_index)) * 0.5;
        let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.5;
        out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
        out.vert_pos = out.clip_position.xyz;
        out.color = vec3<f32>(1.0, 0.0, 0.0); // Red color for the first triangle
    }
    // Define vertices for the second triangle
    else {
        let x = f32(1 - i32(in_vertex_index)) * 0.25;
        let y = f32(i32(in_vertex_index & 1u) * 2 - 1) * 0.25;
        out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
        out.vert_pos = out.clip_position.xyz;
        out.color = vec3<f32>(0.0, 1.0, 0.0); // Green color for the second triangle
    }
    
    return out;
}




@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
