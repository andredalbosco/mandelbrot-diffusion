
const mandelbrot = (() => {
const getTensorBuffer = (safetensorBuffer, tensorMetadata) => {
  return safetensorBuffer.subarray(...tensorMetadata.data_offsets);
};

const getTensorMetadata = (safetensorBuffer) => {
    const metadataLength = Number(new DataView(safetensorBuffer.buffer).getBigUint64(0, true));
    const metadata = JSON.parse(new TextDecoder("utf8").decode(safetensorBuffer.subarray(8, 8 + metadataLength)));
    return Object.fromEntries(Object.entries(metadata).filter(([k, v]) => k !== "__metadata__").map(([k, v]) => [k, {...v, data_offsets: v.data_offsets.map(x => 8 + metadataLength + x)}]));
};

const createEmptyBuf = (device, size) => {
    return device.createBuffer({size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
};

const createUniformBuf = (device, size) => {
  return device.createBuffer({size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST})
}

const createInfinityUniformBuf = (device) => {
  const size = 4;
  const buf = device.createBuffer({
    mappedAtCreation: true,
    size,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
  });
  new Float32Array(buf.getMappedRange())[0] = Infinity;
  buf.unmap();
  return buf;
};

const createWeightBuf = (device, size, data) => {
  const buf = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE, mappedAtCreation: true });
  new Uint8Array(buf.getMappedRange()).set(data); buf.unmap();
  return buf;
};

const addComputePass = (device, commandEncoder, pipeline, layout, infinityUniformBuf, bufs, workgroup) => {
  const bindGroup = device.createBindGroup({
    layout: layout,
    entries: [
      { binding: 0, resource: { buffer: infinityUniformBuf } },
      ...bufs.map((buffer, index) => ({ binding: index + 1, resource: { buffer } }))
    ]
  });

  const passEncoder = commandEncoder.beginComputePass();
  passEncoder.setPipeline(pipeline);
  passEncoder.setBindGroup(0, bindGroup);
  passEncoder.dispatchWorkgroups(...workgroup);
  passEncoder.end();
};

const r_16_2_16_4_4_4_7_8_8_dispatch_0 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_32768:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_49152:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_16384:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_49152:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_57344:array<f32>;
@compute @workgroup_size(2,16,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 2 */
  var lidx1 = i32(lindex.y); /* 16 */
  var lidx2 = i32(lindex.z); /* 4 */
  var cast0 = bitcast<u32>(lidx1);
  var cast1 = bitcast<u32>(lidx2);
  var cast2 = bitcast<i32>((cast1<<5u));
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 7; Lidx0++) {
    var cast3 = bitcast<u32>(Lidx0);
    var alu16 = (Lidx0<3);
    var alu17 = (Lidx0<6);
    var alu18 = ((2<Lidx0)&alu17);
    for (var Lidx1 = 0; Lidx1 < 8; Lidx1++) {
      var cast4 = bitcast<u32>(Lidx1);
      var alu19 = (bitcast<i32>((cast0<<10u))+bitcast<i32>((cast4<<7u)));
      var alu20 = (alu19+bitcast<i32>((cast3<<14u))+cast2);
      var val0 = select(0.0f, data1_49152[alu20], alu16);
      var val1 = select(0.0f, data1_49152[(alu20+1)], alu16);
      var val2 = select(0.0f, data1_49152[(alu20+10)], alu16);
      var val3 = select(0.0f, data1_49152[(alu20+11)], alu16);
      var val4 = select(0.0f, data1_49152[(alu20+12)], alu16);
      var val5 = select(0.0f, data1_49152[(alu20+13)], alu16);
      var val6 = select(0.0f, data1_49152[(alu20+14)], alu16);
      var val7 = select(0.0f, data1_49152[(alu20+15)], alu16);
      var val8 = select(0.0f, data1_49152[(alu20+16)], alu16);
      var val9 = select(0.0f, data1_49152[(alu20+17)], alu16);
      var val10 = select(0.0f, data1_49152[(alu20+18)], alu16);
      var val11 = select(0.0f, data1_49152[(alu20+19)], alu16);
      var val12 = select(0.0f, data1_49152[(alu20+2)], alu16);
      var val13 = select(0.0f, data1_49152[(alu20+20)], alu16);
      var val14 = select(0.0f, data1_49152[(alu20+21)], alu16);
      var val15 = select(0.0f, data1_49152[(alu20+22)], alu16);
      var val16 = select(0.0f, data1_49152[(alu20+23)], alu16);
      var val17 = select(0.0f, data1_49152[(alu20+24)], alu16);
      var val18 = select(0.0f, data1_49152[(alu20+25)], alu16);
      var val19 = select(0.0f, data1_49152[(alu20+26)], alu16);
      var val20 = select(0.0f, data1_49152[(alu20+27)], alu16);
      var val21 = select(0.0f, data1_49152[(alu20+28)], alu16);
      var val22 = select(0.0f, data1_49152[(alu20+29)], alu16);
      var val23 = select(0.0f, data1_49152[(alu20+3)], alu16);
      var val24 = select(0.0f, data1_49152[(alu20+30)], alu16);
      var val25 = select(0.0f, data1_49152[(alu20+31)], alu16);
      var val26 = select(0.0f, data1_49152[(alu20+4)], alu16);
      var val27 = select(0.0f, data1_49152[(alu20+5)], alu16);
      var val28 = select(0.0f, data1_49152[(alu20+6)], alu16);
      var val29 = select(0.0f, data1_49152[(alu20+7)], alu16);
      var val30 = select(0.0f, data1_49152[(alu20+8)], alu16);
      var val31 = select(0.0f, data1_49152[(alu20+9)], alu16);
      var alu21 = (alu19+cast2);
      var val32 = data2_16384[alu21];
      var val33 = data2_16384[(alu21+1)];
      var val34 = data2_16384[(alu21+10)];
      var val35 = data2_16384[(alu21+11)];
      var val36 = data2_16384[(alu21+12)];
      var val37 = data2_16384[(alu21+13)];
      var val38 = data2_16384[(alu21+14)];
      var val39 = data2_16384[(alu21+15)];
      var val40 = data2_16384[(alu21+16)];
      var val41 = data2_16384[(alu21+17)];
      var val42 = data2_16384[(alu21+18)];
      var val43 = data2_16384[(alu21+19)];
      var val44 = data2_16384[(alu21+2)];
      var val45 = data2_16384[(alu21+20)];
      var val46 = data2_16384[(alu21+21)];
      var val47 = data2_16384[(alu21+22)];
      var val48 = data2_16384[(alu21+23)];
      var val49 = data2_16384[(alu21+24)];
      var val50 = data2_16384[(alu21+25)];
      var val51 = data2_16384[(alu21+26)];
      var val52 = data2_16384[(alu21+27)];
      var val53 = data2_16384[(alu21+28)];
      var val54 = data2_16384[(alu21+29)];
      var val55 = data2_16384[(alu21+3)];
      var val56 = data2_16384[(alu21+30)];
      var val57 = data2_16384[(alu21+31)];
      var val58 = data2_16384[(alu21+4)];
      var val59 = data2_16384[(alu21+5)];
      var val60 = data2_16384[(alu21+6)];
      var val61 = data2_16384[(alu21+7)];
      var val62 = data2_16384[(alu21+8)];
      var val63 = data2_16384[(alu21+9)];
      var val64 = select(0.0f, data3_49152[(alu20+i32(-49121))], alu18);
      var val65 = select(0.0f, data3_49152[(alu20+i32(-49122))], alu18);
      var val66 = select(0.0f, data3_49152[(alu20+i32(-49123))], alu18);
      var val67 = select(0.0f, data3_49152[(alu20+i32(-49124))], alu18);
      var val68 = select(0.0f, data3_49152[(alu20+i32(-49125))], alu18);
      var val69 = select(0.0f, data3_49152[(alu20+i32(-49126))], alu18);
      var val70 = select(0.0f, data3_49152[(alu20+i32(-49127))], alu18);
      var val71 = select(0.0f, data3_49152[(alu20+i32(-49128))], alu18);
      var val72 = select(0.0f, data3_49152[(alu20+i32(-49129))], alu18);
      var val73 = select(0.0f, data3_49152[(alu20+i32(-49130))], alu18);
      var val74 = select(0.0f, data3_49152[(alu20+i32(-49131))], alu18);
      var val75 = select(0.0f, data3_49152[(alu20+i32(-49132))], alu18);
      var val76 = select(0.0f, data3_49152[(alu20+i32(-49133))], alu18);
      var val77 = select(0.0f, data3_49152[(alu20+i32(-49134))], alu18);
      var val78 = select(0.0f, data3_49152[(alu20+i32(-49135))], alu18);
      var val79 = select(0.0f, data3_49152[(alu20+i32(-49136))], alu18);
      var val80 = select(0.0f, data3_49152[(alu20+i32(-49137))], alu18);
      var val81 = select(0.0f, data3_49152[(alu20+i32(-49138))], alu18);
      var val82 = select(0.0f, data3_49152[(alu20+i32(-49139))], alu18);
      var val83 = select(0.0f, data3_49152[(alu20+i32(-49140))], alu18);
      var val84 = select(0.0f, data3_49152[(alu20+i32(-49141))], alu18);
      var val85 = select(0.0f, data3_49152[(alu20+i32(-49142))], alu18);
      var val86 = select(0.0f, data3_49152[(alu20+i32(-49143))], alu18);
      var val87 = select(0.0f, data3_49152[(alu20+i32(-49144))], alu18);
      var val88 = select(0.0f, data3_49152[(alu20+i32(-49145))], alu18);
      var val89 = select(0.0f, data3_49152[(alu20+i32(-49146))], alu18);
      var val90 = select(0.0f, data3_49152[(alu20+i32(-49147))], alu18);
      var val91 = select(0.0f, data3_49152[(alu20+i32(-49148))], alu18);
      var val92 = select(0.0f, data3_49152[(alu20+i32(-49149))], alu18);
      var val93 = select(0.0f, data3_49152[(alu20+i32(-49150))], alu18);
      var val94 = select(0.0f, data3_49152[(alu20+i32(-49151))], alu18);
      var val95 = select(0.0f, data3_49152[(alu20+i32(-49152))], alu18);
      var alu22 = ((gidx0*3584)+(lidx0*1792)+bitcast<i32>((cast3<<6u))+bitcast<i32>((cast4<<3u)));
      var val96 = data4_57344[(alu22+1)];
      var val97 = data4_57344[(alu22+1344)];
      var val98 = data4_57344[(alu22+1345)];
      var val99 = data4_57344[(alu22+1346)];
      var val100 = data4_57344[(alu22+1347)];
      var val101 = data4_57344[(alu22+1348)];
      var val102 = data4_57344[(alu22+1349)];
      var val103 = data4_57344[(alu22+1350)];
      var val104 = data4_57344[(alu22+1351)];
      var val105 = data4_57344[(alu22+2)];
      var val106 = data4_57344[(alu22+3)];
      var val107 = data4_57344[(alu22+4)];
      var val108 = data4_57344[(alu22+448)];
      var val109 = data4_57344[(alu22+449)];
      var val110 = data4_57344[(alu22+450)];
      var val111 = data4_57344[(alu22+451)];
      var val112 = data4_57344[(alu22+452)];
      var val113 = data4_57344[(alu22+453)];
      var val114 = data4_57344[(alu22+454)];
      var val115 = data4_57344[(alu22+455)];
      var val116 = data4_57344[(alu22+5)];
      var val117 = data4_57344[(alu22+6)];
      var val118 = data4_57344[(alu22+7)];
      var val119 = data4_57344[(alu22+896)];
      var val120 = data4_57344[(alu22+897)];
      var val121 = data4_57344[(alu22+898)];
      var val122 = data4_57344[(alu22+899)];
      var val123 = data4_57344[(alu22+900)];
      var val124 = data4_57344[(alu22+901)];
      var val125 = data4_57344[(alu22+902)];
      var val126 = data4_57344[(alu22+903)];
      var val127 = data4_57344[alu22];
      var alu23 = select(0.0f,val95,(0.0f<val32));
      var alu24 = select(0.0f,val94,(0.0f<val33));
      var alu25 = select(0.0f,val85,(0.0f<val34));
      var alu26 = select(0.0f,val84,(0.0f<val35));
      var alu27 = select(0.0f,val83,(0.0f<val36));
      var alu28 = select(0.0f,val82,(0.0f<val37));
      var alu29 = select(0.0f,val81,(0.0f<val38));
      var alu30 = select(0.0f,val80,(0.0f<val39));
      var alu31 = select(0.0f,val79,(0.0f<val40));
      var alu32 = select(0.0f,val78,(0.0f<val41));
      var alu33 = select(0.0f,val77,(0.0f<val42));
      var alu34 = select(0.0f,val76,(0.0f<val43));
      var alu35 = select(0.0f,val93,(0.0f<val44));
      var alu36 = select(0.0f,val75,(0.0f<val45));
      var alu37 = select(0.0f,val74,(0.0f<val46));
      var alu38 = select(0.0f,val73,(0.0f<val47));
      var alu39 = select(0.0f,val72,(0.0f<val48));
      var alu40 = select(0.0f,val71,(0.0f<val49));
      var alu41 = select(0.0f,val70,(0.0f<val50));
      var alu42 = select(0.0f,val69,(0.0f<val51));
      var alu43 = select(0.0f,val68,(0.0f<val52));
      var alu44 = select(0.0f,val67,(0.0f<val53));
      var alu45 = select(0.0f,val66,(0.0f<val54));
      var alu46 = select(0.0f,val92,(0.0f<val55));
      var alu47 = select(0.0f,val65,(0.0f<val56));
      var alu48 = select(0.0f,val64,(0.0f<val57));
      var alu49 = select(0.0f,val91,(0.0f<val58));
      var alu50 = select(0.0f,val90,(0.0f<val59));
      var alu51 = select(0.0f,val89,(0.0f<val60));
      var alu52 = select(0.0f,val88,(0.0f<val61));
      var alu53 = select(0.0f,val87,(0.0f<val62));
      var alu54 = select(0.0f,val86,(0.0f<val63));
      var alu55 = select(val32,0.0f,alu17);
      var alu56 = (val0+alu23+alu55);
      var alu57 = select(val33,0.0f,alu17);
      var alu58 = (val1+alu24+alu57);
      var alu59 = select(val44,0.0f,alu17);
      var alu60 = (val12+alu35+alu59);
      var alu61 = select(val55,0.0f,alu17);
      var alu62 = (val23+alu46+alu61);
      var alu63 = select(val58,0.0f,alu17);
      var alu64 = (val26+alu49+alu63);
      var alu65 = select(val59,0.0f,alu17);
      var alu66 = (val27+alu50+alu65);
      var alu67 = select(val60,0.0f,alu17);
      var alu68 = (val28+alu51+alu67);
      var alu69 = select(val61,0.0f,alu17);
      var alu70 = (val29+alu52+alu69);
      var alu71 = select(val34,0.0f,alu17);
      var alu72 = (val2+alu25+alu71);
      var alu73 = select(val35,0.0f,alu17);
      var alu74 = (val3+alu26+alu73);
      var alu75 = select(val36,0.0f,alu17);
      var alu76 = (val4+alu27+alu75);
      var alu77 = select(val37,0.0f,alu17);
      var alu78 = (val5+alu28+alu77);
      var alu79 = select(val38,0.0f,alu17);
      var alu80 = (val6+alu29+alu79);
      var alu81 = select(val39,0.0f,alu17);
      var alu82 = (val7+alu30+alu81);
      var alu83 = select(val62,0.0f,alu17);
      var alu84 = (val30+alu53+alu83);
      var alu85 = select(val63,0.0f,alu17);
      var alu86 = (val31+alu54+alu85);
      var alu87 = select(val40,0.0f,alu17);
      var alu88 = (val8+alu31+alu87);
      var alu89 = select(val41,0.0f,alu17);
      var alu90 = (val9+alu32+alu89);
      var alu91 = select(val42,0.0f,alu17);
      var alu92 = (val10+alu33+alu91);
      var alu93 = select(val43,0.0f,alu17);
      var alu94 = (val11+alu34+alu93);
      var alu95 = select(val45,0.0f,alu17);
      var alu96 = (val13+alu36+alu95);
      var alu97 = select(val46,0.0f,alu17);
      var alu98 = (val14+alu37+alu97);
      var alu99 = select(val47,0.0f,alu17);
      var alu100 = (val15+alu38+alu99);
      var alu101 = select(val48,0.0f,alu17);
      var alu102 = (val16+alu39+alu101);
      var alu103 = select(val49,0.0f,alu17);
      var alu104 = (val17+alu40+alu103);
      var alu105 = select(val50,0.0f,alu17);
      var alu106 = (val18+alu41+alu105);
      var alu107 = select(val51,0.0f,alu17);
      var alu108 = (val19+alu42+alu107);
      var alu109 = select(val52,0.0f,alu17);
      var alu110 = (val20+alu43+alu109);
      var alu111 = select(val53,0.0f,alu17);
      var alu112 = (val21+alu44+alu111);
      var alu113 = select(val54,0.0f,alu17);
      var alu114 = (val22+alu45+alu113);
      var alu115 = select(val56,0.0f,alu17);
      var alu116 = (val24+alu47+alu115);
      var alu117 = select(val57,0.0f,alu17);
      var alu118 = (val25+alu48+alu117);
      buf0[0] = (buf0[0]+(alu56*val127)+(alu58*val96)+(alu60*val105)+(alu62*val106)+(alu64*val107)+(alu66*val116)+(alu68*val117)+(alu70*val118));
      buf0[1] = (buf0[1]+(alu84*val127)+(alu86*val96)+(alu72*val105)+(alu74*val106)+(alu76*val107)+(alu78*val116)+(alu80*val117)+(alu82*val118));
      buf0[10] = (buf0[10]+(alu88*val119)+(alu90*val120)+(alu92*val121)+(alu94*val122)+(alu96*val123)+(alu98*val124)+(alu100*val125)+(alu102*val126));
      buf0[11] = (buf0[11]+(alu104*val119)+(alu106*val120)+(alu108*val121)+(alu110*val122)+(alu112*val123)+(alu114*val124)+(alu116*val125)+(alu118*val126));
      buf0[12] = (buf0[12]+(alu56*val97)+(alu58*val98)+(alu60*val99)+(alu62*val100)+(alu64*val101)+(alu66*val102)+(alu68*val103)+(alu70*val104));
      buf0[13] = (buf0[13]+(alu84*val97)+(alu86*val98)+(alu72*val99)+(alu74*val100)+(alu76*val101)+(alu78*val102)+(alu80*val103)+(alu82*val104));
      buf0[14] = (buf0[14]+(alu88*val97)+(alu90*val98)+(alu92*val99)+(alu94*val100)+(alu96*val101)+(alu98*val102)+(alu100*val103)+(alu102*val104));
      buf0[15] = (buf0[15]+(alu104*val97)+(alu106*val98)+(alu108*val99)+(alu110*val100)+(alu112*val101)+(alu114*val102)+(alu116*val103)+(alu118*val104));
      buf0[2] = (buf0[2]+(alu88*val127)+(alu90*val96)+(alu92*val105)+(alu94*val106)+(alu96*val107)+(alu98*val116)+(alu100*val117)+(alu102*val118));
      buf0[3] = (buf0[3]+(alu104*val127)+(alu106*val96)+(alu108*val105)+(alu110*val106)+(alu112*val107)+(alu114*val116)+(alu116*val117)+(alu118*val118));
      buf0[4] = (buf0[4]+(alu56*val108)+(alu58*val109)+(alu60*val110)+(alu62*val111)+(alu64*val112)+(alu66*val113)+(alu68*val114)+(alu70*val115));
      buf0[5] = (buf0[5]+(alu84*val108)+(alu86*val109)+(alu72*val110)+(alu74*val111)+(alu76*val112)+(alu78*val113)+(alu80*val114)+(alu82*val115));
      buf0[6] = (buf0[6]+(alu88*val108)+(alu90*val109)+(alu92*val110)+(alu94*val111)+(alu96*val112)+(alu98*val113)+(alu100*val114)+(alu102*val115));
      buf0[7] = (buf0[7]+(alu104*val108)+(alu106*val109)+(alu108*val110)+(alu110*val111)+(alu112*val112)+(alu114*val113)+(alu116*val114)+(alu118*val115));
      buf0[8] = (buf0[8]+(alu56*val119)+(alu58*val120)+(alu60*val121)+(alu62*val122)+(alu64*val123)+(alu66*val124)+(alu68*val125)+(alu70*val126));
      buf0[9] = (buf0[9]+(alu84*val119)+(alu86*val120)+(alu72*val121)+(alu74*val122)+(alu76*val123)+(alu78*val124)+(alu80*val125)+(alu82*val126));
    }
  }
  var alu137 = (bitcast<i32>((bitcast<u32>(gidx0)<<11u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+bitcast<i32>((cast0<<4u))+bitcast<i32>((cast1<<2u)));
  data0_32768[alu137] = buf0[0];
  data0_32768[(alu137+1)] = buf0[1];
  data0_32768[(alu137+2)] = buf0[2];
  data0_32768[(alu137+256)] = buf0[4];
  data0_32768[(alu137+257)] = buf0[5];
  data0_32768[(alu137+258)] = buf0[6];
  data0_32768[(alu137+259)] = buf0[7];
  data0_32768[(alu137+3)] = buf0[3];
  data0_32768[(alu137+512)] = buf0[8];
  data0_32768[(alu137+513)] = buf0[9];
  data0_32768[(alu137+514)] = buf0[10];
  data0_32768[(alu137+515)] = buf0[11];
  data0_32768[(alu137+768)] = buf0[12];
  data0_32768[(alu137+769)] = buf0[13];
  data0_32768[(alu137+770)] = buf0[14];
  data0_32768[(alu137+771)] = buf0[15];
}`;

const r_256_16_2_8_dispatch_1 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_1:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_128:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_256:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var val0 = data1_1[0];
  var gidx0 = i32(gindex.x); /* 256 */
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = 0.0f;
  for (var Lidx0_0 = 0; Lidx0_0 < 2; Lidx0_0++) {
    for (var Lidx0_1 = 0; Lidx0_1 < 8; Lidx0_1++) {
      var alu1 = (bitcast<i32>((bitcast<u32>(lidx0)<<3u))+Lidx0_1);
      var val1 = data2_128[alu1];
      var val2 = data3_65536[(alu1+bitcast<i32>((bitcast<u32>(Lidx0_0)<<7u))+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
      var alu2 = (val0*val1);
      var alu3 = select(sin(alu2),sin((1.5707963705062866f-alu2)),(Lidx0_0<1));
      buf0[0] = (buf0[0]+(alu3*val2));
    }
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val3 = buf2[Lidx102];
    buf1[0] = (buf1[0]+val3);
  }
  var val4 = data4_256[gidx0];
  var alu12 = (buf1[0]+val4);
  if ((lidx0==0)) {
    data0_256[gidx0] = (alu12*(1/(1.0f+exp2((alu12*-1.4426950216293335f)))));
  }
}`;

const r_8_4_8_16_4_4_32_4_dispatch_2 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_32768:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_32768:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_65536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var cast4 = bitcast<u32>(Lidx0);
    var alu16 = (bitcast<i32>((cast1<<5u))+bitcast<i32>((cast2<<2u))+bitcast<i32>((cast4<<10u)));
    var val0 = data1_32768[alu16];
    var val1 = data1_32768[(alu16+1)];
    var val2 = data1_32768[(alu16+2)];
    var val3 = data1_32768[(alu16+256)];
    var val4 = data1_32768[(alu16+257)];
    var val5 = data1_32768[(alu16+258)];
    var val6 = data1_32768[(alu16+259)];
    var val7 = data1_32768[(alu16+3)];
    var val8 = data1_32768[(alu16+512)];
    var val9 = data1_32768[(alu16+513)];
    var val10 = data1_32768[(alu16+514)];
    var val11 = data1_32768[(alu16+515)];
    var val12 = data1_32768[(alu16+768)];
    var val13 = data1_32768[(alu16+769)];
    var val14 = data1_32768[(alu16+770)];
    var val15 = data1_32768[(alu16+771)];
    var alu17 = (bitcast<i32>((cast0<<13u))+bitcast<i32>((cast3<<9u))+bitcast<i32>((cast4<<2u)));
    var val16 = data2_32768[alu17];
    var val17 = data2_32768[(alu17+1)];
    var val18 = data2_32768[(alu17+128)];
    var val19 = data2_32768[(alu17+129)];
    var val20 = data2_32768[(alu17+130)];
    var val21 = data2_32768[(alu17+131)];
    var val22 = data2_32768[(alu17+2)];
    var val23 = data2_32768[(alu17+256)];
    var val24 = data2_32768[(alu17+257)];
    var val25 = data2_32768[(alu17+258)];
    var val26 = data2_32768[(alu17+259)];
    var val27 = data2_32768[(alu17+3)];
    var val28 = data2_32768[(alu17+384)];
    var val29 = data2_32768[(alu17+385)];
    var val30 = data2_32768[(alu17+386)];
    var val31 = data2_32768[(alu17+387)];
    buf0[0] = (buf0[0]+(val0*val16)+(val3*val17)+(val8*val22)+(val12*val27));
    buf0[1] = (buf0[1]+(val0*val18)+(val3*val19)+(val8*val20)+(val12*val21));
    buf0[10] = (buf0[10]+(val2*val23)+(val5*val24)+(val10*val25)+(val14*val26));
    buf0[11] = (buf0[11]+(val2*val28)+(val5*val29)+(val10*val30)+(val14*val31));
    buf0[12] = (buf0[12]+(val7*val16)+(val6*val17)+(val11*val22)+(val15*val27));
    buf0[13] = (buf0[13]+(val7*val18)+(val6*val19)+(val11*val20)+(val15*val21));
    buf0[14] = (buf0[14]+(val7*val23)+(val6*val24)+(val11*val25)+(val15*val26));
    buf0[15] = (buf0[15]+(val7*val28)+(val6*val29)+(val11*val30)+(val15*val31));
    buf0[2] = (buf0[2]+(val0*val23)+(val3*val24)+(val8*val25)+(val12*val26));
    buf0[3] = (buf0[3]+(val0*val28)+(val3*val29)+(val8*val30)+(val12*val31));
    buf0[4] = (buf0[4]+(val1*val16)+(val4*val17)+(val9*val22)+(val13*val27));
    buf0[5] = (buf0[5]+(val1*val18)+(val4*val19)+(val9*val20)+(val13*val21));
    buf0[6] = (buf0[6]+(val1*val23)+(val4*val24)+(val9*val25)+(val13*val26));
    buf0[7] = (buf0[7]+(val1*val28)+(val4*val29)+(val9*val30)+(val13*val31));
    buf0[8] = (buf0[8]+(val2*val16)+(val5*val17)+(val10*val22)+(val14*val27));
    buf0[9] = (buf0[9]+(val2*val18)+(val5*val19)+(val10*val20)+(val14*val21));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var val32 = data3_256[alu35];
  var val33 = data3_256[(alu35+1)];
  var val34 = data3_256[(alu35+2)];
  var val35 = data3_256[(alu35+3)];
  var alu36 = (alu35+bitcast<i32>((cast1<<13u))+bitcast<i32>((cast2<<10u)));
  var val36 = data4_65536[alu36];
  var alu37 = (alu36+1);
  var val37 = data4_65536[alu37];
  var alu38 = (alu36+2);
  var val38 = data4_65536[alu38];
  var alu39 = (alu36+256);
  var val39 = data4_65536[alu39];
  var alu40 = (alu36+257);
  var val40 = data4_65536[alu40];
  var alu41 = (alu36+258);
  var val41 = data4_65536[alu41];
  var alu42 = (alu36+259);
  var val42 = data4_65536[alu42];
  var alu43 = (alu36+3);
  var val43 = data4_65536[alu43];
  var alu44 = (alu36+512);
  var val44 = data4_65536[alu44];
  var alu45 = (alu36+513);
  var val45 = data4_65536[alu45];
  var alu46 = (alu36+514);
  var val46 = data4_65536[alu46];
  var alu47 = (alu36+515);
  var val47 = data4_65536[alu47];
  var alu48 = (alu36+768);
  var val48 = data4_65536[alu48];
  var alu49 = (alu36+769);
  var val49 = data4_65536[alu49];
  var alu50 = (alu36+770);
  var val50 = data4_65536[alu50];
  var alu51 = (alu36+771);
  var val51 = data4_65536[alu51];
  data0_65536[alu36] = (buf0[0]+val32+val36);
  data0_65536[alu37] = (buf0[1]+val33+val37);
  data0_65536[alu38] = (buf0[2]+val34+val38);
  data0_65536[alu39] = (buf0[4]+val32+val39);
  data0_65536[alu40] = (buf0[5]+val33+val40);
  data0_65536[alu41] = (buf0[6]+val34+val41);
  data0_65536[alu42] = (buf0[7]+val35+val42);
  data0_65536[alu43] = (buf0[3]+val35+val43);
  data0_65536[alu44] = (buf0[8]+val32+val44);
  data0_65536[alu45] = (buf0[9]+val33+val45);
  data0_65536[alu46] = (buf0[10]+val34+val46);
  data0_65536[alu47] = (buf0[11]+val35+val47);
  data0_65536[alu48] = (buf0[12]+val32+val48);
  data0_65536[alu49] = (buf0[13]+val33+val49);
  data0_65536[alu50] = (buf0[14]+val34+val50);
  data0_65536[alu51] = (buf0[15]+val35+val51);
}`;

const r_16_8_4_4_32_dispatch_3 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_256:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_256[alu4];
    var alu5 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+alu4);
    var val1 = data2_65536[alu5];
    var val2 = data2_65536[(alu5+1024)];
    var val3 = data2_65536[(alu5+2048)];
    var val4 = data2_65536[(alu5+3072)];
    buf0[0] = (buf0[0]+(val0*val1));
    buf0[1] = (buf0[1]+(val0*val2));
    buf0[2] = (buf0[2]+(val0*val3));
    buf0[3] = (buf0[3]+(val0*val4));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu11 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu11] = buf0[0];
  buf2[(alu11+1)] = buf0[1];
  buf2[(alu11+2)] = buf0[2];
  buf2[(alu11+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu21 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val5 = buf2[alu21];
    var val6 = buf2[(alu21+1)];
    var val7 = buf2[(alu21+2)];
    var val8 = buf2[(alu21+3)];
    buf1[0] = (buf1[0]+val5);
    buf1[1] = (buf1[1]+val6);
    buf1[2] = (buf1[2]+val7);
    buf1[3] = (buf1[3]+val8);
  }
  var alu27 = (bitcast<i32>((cast0<<4u))+lidx1);
  var val9 = data3_256[alu27];
  var alu28 = (alu27+12);
  var val10 = data3_256[alu28];
  var alu29 = (alu27+4);
  var val11 = data3_256[alu29];
  var alu30 = (alu27+8);
  var val12 = data3_256[alu30];
  var alu31 = (buf1[0]+val9);
  var alu32 = (buf1[1]+val11);
  var alu33 = (buf1[2]+val12);
  var alu34 = (buf1[3]+val10);
  var alu35 = (lidx0==0);
  if (alu35) {
    data0_256[alu27] = (alu31*(1/(1.0f+exp2((alu31*-1.4426950216293335f)))));
  }
  if (alu35) {
    data0_256[alu28] = (alu34*(1/(1.0f+exp2((alu34*-1.4426950216293335f)))));
  }
  if (alu35) {
    data0_256[alu29] = (alu32*(1/(1.0f+exp2((alu32*-1.4426950216293335f)))));
  }
  if (alu35) {
    data0_256[alu30] = (alu33*(1/(1.0f+exp2((alu33*-1.4426950216293335f)))));
  }
}`;

const r_16_8_4_4_32_dispatch_4 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const r_96_8_4_4_32_dispatch_5 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_1536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_256:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_393216:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 96 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_256[alu4];
    var alu5 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+alu4);
    var val1 = data2_393216[alu5];
    var val2 = data2_393216[(alu5+1024)];
    var val3 = data2_393216[(alu5+2048)];
    var val4 = data2_393216[(alu5+3072)];
    buf0[0] = (buf0[0]+(val0*val1));
    buf0[1] = (buf0[1]+(val0*val2));
    buf0[2] = (buf0[2]+(val0*val3));
    buf0[3] = (buf0[3]+(val0*val4));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu11 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu11] = buf0[0];
  buf2[(alu11+1)] = buf0[1];
  buf2[(alu11+2)] = buf0[2];
  buf2[(alu11+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu21 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val5 = buf2[alu21];
    var val6 = buf2[(alu21+1)];
    var val7 = buf2[(alu21+2)];
    var val8 = buf2[(alu21+3)];
    buf1[0] = (buf1[0]+val5);
    buf1[1] = (buf1[1]+val6);
    buf1[2] = (buf1[2]+val7);
    buf1[3] = (buf1[3]+val8);
  }
  var alu27 = (bitcast<i32>((cast0<<4u))+lidx1);
  var val9 = data3_1536[alu27];
  var alu28 = (alu27+12);
  var val10 = data3_1536[alu28];
  var alu29 = (alu27+4);
  var val11 = data3_1536[alu29];
  var alu30 = (alu27+8);
  var val12 = data3_1536[alu30];
  var alu31 = (lidx0==0);
  if (alu31) {
    data0_1536[alu27] = (buf1[0]+val9);
  }
  if (alu31) {
    data0_1536[alu28] = (buf1[3]+val10);
  }
  if (alu31) {
    data0_1536[alu29] = (buf1[1]+val11);
  }
  if (alu31) {
    data0_1536[alu30] = (buf1[2]+val12);
  }
}`;

const r_96_8_4_4_32_dispatch_6 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_1536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_256:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_393216:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 96 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_256[alu4];
    var alu5 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+alu4);
    var val1 = data2_393216[alu5];
    var val2 = data2_393216[(alu5+1024)];
    var val3 = data2_393216[(alu5+2048)];
    var val4 = data2_393216[(alu5+3072)];
    buf0[0] = (buf0[0]+(val0*val1));
    buf0[1] = (buf0[1]+(val0*val2));
    buf0[2] = (buf0[2]+(val0*val3));
    buf0[3] = (buf0[3]+(val0*val4));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu11 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu11] = buf0[0];
  buf2[(alu11+1)] = buf0[1];
  buf2[(alu11+2)] = buf0[2];
  buf2[(alu11+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu21 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val5 = buf2[alu21];
    var val6 = buf2[(alu21+1)];
    var val7 = buf2[(alu21+2)];
    var val8 = buf2[(alu21+3)];
    buf1[0] = (buf1[0]+val5);
    buf1[1] = (buf1[1]+val6);
    buf1[2] = (buf1[2]+val7);
    buf1[3] = (buf1[3]+val8);
  }
  var alu27 = (bitcast<i32>((cast0<<4u))+lidx1);
  var val9 = data3_1536[alu27];
  var alu28 = (alu27+12);
  var val10 = data3_1536[alu28];
  var alu29 = (alu27+4);
  var val11 = data3_1536[alu29];
  var alu30 = (alu27+8);
  var val12 = data3_1536[alu30];
  var alu31 = (lidx0==0);
  if (alu31) {
    data0_1536[alu27] = (buf1[0]+val9);
  }
  if (alu31) {
    data0_1536[alu28] = (buf1[3]+val10);
  }
  if (alu31) {
    data0_1536[alu29] = (buf1[1]+val11);
  }
  if (alu31) {
    data0_1536[alu30] = (buf1[2]+val12);
  }
}`;

const r_96_8_4_4_32_dispatch_7 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_1536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_256:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_393216:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 96 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_256[alu4];
    var alu5 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+alu4);
    var val1 = data2_393216[alu5];
    var val2 = data2_393216[(alu5+1024)];
    var val3 = data2_393216[(alu5+2048)];
    var val4 = data2_393216[(alu5+3072)];
    buf0[0] = (buf0[0]+(val0*val1));
    buf0[1] = (buf0[1]+(val0*val2));
    buf0[2] = (buf0[2]+(val0*val3));
    buf0[3] = (buf0[3]+(val0*val4));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu11 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu11] = buf0[0];
  buf2[(alu11+1)] = buf0[1];
  buf2[(alu11+2)] = buf0[2];
  buf2[(alu11+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu21 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val5 = buf2[alu21];
    var val6 = buf2[(alu21+1)];
    var val7 = buf2[(alu21+2)];
    var val8 = buf2[(alu21+3)];
    buf1[0] = (buf1[0]+val5);
    buf1[1] = (buf1[1]+val6);
    buf1[2] = (buf1[2]+val7);
    buf1[3] = (buf1[3]+val8);
  }
  var alu27 = (bitcast<i32>((cast0<<4u))+lidx1);
  var val9 = data3_1536[alu27];
  var alu28 = (alu27+12);
  var val10 = data3_1536[alu28];
  var alu29 = (alu27+4);
  var val11 = data3_1536[alu29];
  var alu30 = (alu27+8);
  var val12 = data3_1536[alu30];
  var alu31 = (lidx0==0);
  if (alu31) {
    data0_1536[alu27] = (buf1[0]+val9);
  }
  if (alu31) {
    data0_1536[alu28] = (buf1[3]+val10);
  }
  if (alu31) {
    data0_1536[alu29] = (buf1[1]+val11);
  }
  if (alu31) {
    data0_1536[alu30] = (buf1[2]+val12);
  }
}`;

const r_96_8_4_4_32_dispatch_8 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_1536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_256:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_393216:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 96 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_256[alu4];
    var alu5 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+alu4);
    var val1 = data2_393216[alu5];
    var val2 = data2_393216[(alu5+1024)];
    var val3 = data2_393216[(alu5+2048)];
    var val4 = data2_393216[(alu5+3072)];
    buf0[0] = (buf0[0]+(val0*val1));
    buf0[1] = (buf0[1]+(val0*val2));
    buf0[2] = (buf0[2]+(val0*val3));
    buf0[3] = (buf0[3]+(val0*val4));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu11 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu11] = buf0[0];
  buf2[(alu11+1)] = buf0[1];
  buf2[(alu11+2)] = buf0[2];
  buf2[(alu11+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu21 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val5 = buf2[alu21];
    var val6 = buf2[(alu21+1)];
    var val7 = buf2[(alu21+2)];
    var val8 = buf2[(alu21+3)];
    buf1[0] = (buf1[0]+val5);
    buf1[1] = (buf1[1]+val6);
    buf1[2] = (buf1[2]+val7);
    buf1[3] = (buf1[3]+val8);
  }
  var alu27 = (bitcast<i32>((cast0<<4u))+lidx1);
  var val9 = data3_1536[alu27];
  var alu28 = (alu27+12);
  var val10 = data3_1536[alu28];
  var alu29 = (alu27+4);
  var val11 = data3_1536[alu29];
  var alu30 = (alu27+8);
  var val12 = data3_1536[alu30];
  var alu31 = (lidx0==0);
  if (alu31) {
    data0_1536[alu27] = (buf1[0]+val9);
  }
  if (alu31) {
    data0_1536[alu28] = (buf1[3]+val10);
  }
  if (alu31) {
    data0_1536[alu29] = (buf1[1]+val11);
  }
  if (alu31) {
    data0_1536[alu30] = (buf1[2]+val12);
  }
}`;

const r_96_8_4_4_32_dispatch_9 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_1536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_256:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_393216:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 96 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_256[alu4];
    var alu5 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+alu4);
    var val1 = data2_393216[alu5];
    var val2 = data2_393216[(alu5+1024)];
    var val3 = data2_393216[(alu5+2048)];
    var val4 = data2_393216[(alu5+3072)];
    buf0[0] = (buf0[0]+(val0*val1));
    buf0[1] = (buf0[1]+(val0*val2));
    buf0[2] = (buf0[2]+(val0*val3));
    buf0[3] = (buf0[3]+(val0*val4));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu11 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu11] = buf0[0];
  buf2[(alu11+1)] = buf0[1];
  buf2[(alu11+2)] = buf0[2];
  buf2[(alu11+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu21 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val5 = buf2[alu21];
    var val6 = buf2[(alu21+1)];
    var val7 = buf2[(alu21+2)];
    var val8 = buf2[(alu21+3)];
    buf1[0] = (buf1[0]+val5);
    buf1[1] = (buf1[1]+val6);
    buf1[2] = (buf1[2]+val7);
    buf1[3] = (buf1[3]+val8);
  }
  var alu27 = (bitcast<i32>((cast0<<4u))+lidx1);
  var val9 = data3_1536[alu27];
  var alu28 = (alu27+12);
  var val10 = data3_1536[alu28];
  var alu29 = (alu27+4);
  var val11 = data3_1536[alu29];
  var alu30 = (alu27+8);
  var val12 = data3_1536[alu30];
  var alu31 = (lidx0==0);
  if (alu31) {
    data0_1536[alu27] = (buf1[0]+val9);
  }
  if (alu31) {
    data0_1536[alu28] = (buf1[3]+val10);
  }
  if (alu31) {
    data0_1536[alu29] = (buf1[1]+val11);
  }
  if (alu31) {
    data0_1536[alu30] = (buf1[2]+val12);
  }
}`;

const r_96_8_4_4_32_dispatch_10 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_1536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_256:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_393216:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 96 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_256[alu4];
    var alu5 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+alu4);
    var val1 = data2_393216[alu5];
    var val2 = data2_393216[(alu5+1024)];
    var val3 = data2_393216[(alu5+2048)];
    var val4 = data2_393216[(alu5+3072)];
    buf0[0] = (buf0[0]+(val0*val1));
    buf0[1] = (buf0[1]+(val0*val2));
    buf0[2] = (buf0[2]+(val0*val3));
    buf0[3] = (buf0[3]+(val0*val4));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu11 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu11] = buf0[0];
  buf2[(alu11+1)] = buf0[1];
  buf2[(alu11+2)] = buf0[2];
  buf2[(alu11+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu21 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val5 = buf2[alu21];
    var val6 = buf2[(alu21+1)];
    var val7 = buf2[(alu21+2)];
    var val8 = buf2[(alu21+3)];
    buf1[0] = (buf1[0]+val5);
    buf1[1] = (buf1[1]+val6);
    buf1[2] = (buf1[2]+val7);
    buf1[3] = (buf1[3]+val8);
  }
  var alu27 = (bitcast<i32>((cast0<<4u))+lidx1);
  var val9 = data3_1536[alu27];
  var alu28 = (alu27+12);
  var val10 = data3_1536[alu28];
  var alu29 = (alu27+4);
  var val11 = data3_1536[alu29];
  var alu30 = (alu27+8);
  var val12 = data3_1536[alu30];
  var alu31 = (lidx0==0);
  if (alu31) {
    data0_1536[alu27] = (buf1[0]+val9);
  }
  if (alu31) {
    data0_1536[alu28] = (buf1[3]+val10);
  }
  if (alu31) {
    data0_1536[alu29] = (buf1[1]+val11);
  }
  if (alu31) {
    data0_1536[alu30] = (buf1[2]+val12);
  }
}`;

const r_96_8_4_4_32_dispatch_11 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_1536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_256:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_393216:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 96 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_256[alu4];
    var alu5 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+alu4);
    var val1 = data2_393216[alu5];
    var val2 = data2_393216[(alu5+1024)];
    var val3 = data2_393216[(alu5+2048)];
    var val4 = data2_393216[(alu5+3072)];
    buf0[0] = (buf0[0]+(val0*val1));
    buf0[1] = (buf0[1]+(val0*val2));
    buf0[2] = (buf0[2]+(val0*val3));
    buf0[3] = (buf0[3]+(val0*val4));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu11 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu11] = buf0[0];
  buf2[(alu11+1)] = buf0[1];
  buf2[(alu11+2)] = buf0[2];
  buf2[(alu11+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu21 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val5 = buf2[alu21];
    var val6 = buf2[(alu21+1)];
    var val7 = buf2[(alu21+2)];
    var val8 = buf2[(alu21+3)];
    buf1[0] = (buf1[0]+val5);
    buf1[1] = (buf1[1]+val6);
    buf1[2] = (buf1[2]+val7);
    buf1[3] = (buf1[3]+val8);
  }
  var alu27 = (bitcast<i32>((cast0<<4u))+lidx1);
  var val9 = data3_1536[alu27];
  var alu28 = (alu27+12);
  var val10 = data3_1536[alu28];
  var alu29 = (alu27+4);
  var val11 = data3_1536[alu29];
  var alu30 = (alu27+8);
  var val12 = data3_1536[alu30];
  var alu31 = (lidx0==0);
  if (alu31) {
    data0_1536[alu27] = (buf1[0]+val9);
  }
  if (alu31) {
    data0_1536[alu28] = (buf1[3]+val10);
  }
  if (alu31) {
    data0_1536[alu29] = (buf1[1]+val11);
  }
  if (alu31) {
    data0_1536[alu30] = (buf1[2]+val12);
  }
}`;

const r_96_8_4_4_32_dispatch_12 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_1536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_256:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_393216:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 96 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_256[alu4];
    var alu5 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+alu4);
    var val1 = data2_393216[alu5];
    var val2 = data2_393216[(alu5+1024)];
    var val3 = data2_393216[(alu5+2048)];
    var val4 = data2_393216[(alu5+3072)];
    buf0[0] = (buf0[0]+(val0*val1));
    buf0[1] = (buf0[1]+(val0*val2));
    buf0[2] = (buf0[2]+(val0*val3));
    buf0[3] = (buf0[3]+(val0*val4));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu11 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu11] = buf0[0];
  buf2[(alu11+1)] = buf0[1];
  buf2[(alu11+2)] = buf0[2];
  buf2[(alu11+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu21 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val5 = buf2[alu21];
    var val6 = buf2[(alu21+1)];
    var val7 = buf2[(alu21+2)];
    var val8 = buf2[(alu21+3)];
    buf1[0] = (buf1[0]+val5);
    buf1[1] = (buf1[1]+val6);
    buf1[2] = (buf1[2]+val7);
    buf1[3] = (buf1[3]+val8);
  }
  var alu27 = (bitcast<i32>((cast0<<4u))+lidx1);
  var val9 = data3_1536[alu27];
  var alu28 = (alu27+12);
  var val10 = data3_1536[alu28];
  var alu29 = (alu27+4);
  var val11 = data3_1536[alu29];
  var alu30 = (alu27+8);
  var val12 = data3_1536[alu30];
  var alu31 = (lidx0==0);
  if (alu31) {
    data0_1536[alu27] = (buf1[0]+val9);
  }
  if (alu31) {
    data0_1536[alu28] = (buf1[3]+val10);
  }
  if (alu31) {
    data0_1536[alu29] = (buf1[1]+val11);
  }
  if (alu31) {
    data0_1536[alu30] = (buf1[2]+val12);
  }
}`;

const r_32_8_4_4_32_dispatch_13 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_512:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_256:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_131072:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_512:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 32 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_256[alu4];
    var alu5 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+alu4);
    var val1 = data2_131072[alu5];
    var val2 = data2_131072[(alu5+1024)];
    var val3 = data2_131072[(alu5+2048)];
    var val4 = data2_131072[(alu5+3072)];
    buf0[0] = (buf0[0]+(val0*val1));
    buf0[1] = (buf0[1]+(val0*val2));
    buf0[2] = (buf0[2]+(val0*val3));
    buf0[3] = (buf0[3]+(val0*val4));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu11 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu11] = buf0[0];
  buf2[(alu11+1)] = buf0[1];
  buf2[(alu11+2)] = buf0[2];
  buf2[(alu11+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu21 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val5 = buf2[alu21];
    var val6 = buf2[(alu21+1)];
    var val7 = buf2[(alu21+2)];
    var val8 = buf2[(alu21+3)];
    buf1[0] = (buf1[0]+val5);
    buf1[1] = (buf1[1]+val6);
    buf1[2] = (buf1[2]+val7);
    buf1[3] = (buf1[3]+val8);
  }
  var alu27 = (bitcast<i32>((cast0<<4u))+lidx1);
  var val9 = data3_512[alu27];
  var alu28 = (alu27+12);
  var val10 = data3_512[alu28];
  var alu29 = (alu27+4);
  var val11 = data3_512[alu29];
  var alu30 = (alu27+8);
  var val12 = data3_512[alu30];
  var alu31 = (lidx0==0);
  if (alu31) {
    data0_512[alu27] = (buf1[0]+val9);
  }
  if (alu31) {
    data0_512[alu28] = (buf1[3]+val10);
  }
  if (alu31) {
    data0_512[alu29] = (buf1[1]+val11);
  }
  if (alu31) {
    data0_512[alu30] = (buf1[2]+val12);
  }
}`;

const E_8_4_8_16_4_4_dispatch_14 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var alu18 = (alu0+1);
  var val21 = data3_256[alu18];
  var alu19 = (alu0+2);
  var val22 = data3_256[alu19];
  var alu20 = (alu0+3);
  var val23 = data3_256[alu20];
  var val24 = data4_1536[alu0];
  var val25 = data4_1536[alu18];
  var val26 = data4_1536[alu19];
  var val27 = data4_1536[(alu0+256)];
  var val28 = data4_1536[(alu0+257)];
  var val29 = data4_1536[(alu0+258)];
  var val30 = data4_1536[(alu0+259)];
  var val31 = data4_1536[alu20];
  var alu21 = (1/val16);
  var alu22 = (1/val17);
  var alu23 = (1/val18);
  var alu24 = (1/val19);
  var alu25 = (1.0f+val27);
  var alu26 = (1.0f+val28);
  var alu27 = (1.0f+val29);
  var alu28 = (1.0f+val30);
  data0_65536[alu1] = ((val0*alu21*val20*alu25)+val24);
  data0_65536[alu2] = ((val1*alu21*val21*alu26)+val25);
  data0_65536[alu3] = ((val2*alu21*val22*alu27)+val26);
  data0_65536[alu4] = ((val3*alu22*val20*alu25)+val24);
  data0_65536[alu5] = ((val4*alu22*val21*alu26)+val25);
  data0_65536[alu6] = ((val5*alu22*val22*alu27)+val26);
  data0_65536[alu7] = ((val6*alu22*val23*alu28)+val31);
  data0_65536[alu8] = ((val7*alu21*val23*alu28)+val31);
  data0_65536[alu9] = ((val8*alu23*val20*alu25)+val24);
  data0_65536[alu10] = ((val9*alu23*val21*alu26)+val25);
  data0_65536[alu11] = ((val10*alu23*val22*alu27)+val26);
  data0_65536[alu12] = ((val11*alu23*val23*alu28)+val31);
  data0_65536[alu13] = ((val12*alu24*val20*alu25)+val24);
  data0_65536[alu14] = ((val13*alu24*val21*alu26)+val25);
  data0_65536[alu15] = ((val14*alu24*val22*alu27)+val26);
  data0_65536[alu16] = ((val15*alu24*val23*alu28)+val31);
}`;

const r_8_16_8_16_3_4_64_4_dispatch_15 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_196608:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu12 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast0);
    var val0 = data1_65536[alu12];
    var val1 = data1_65536[(alu12+1)];
    var val2 = data1_65536[(alu12+2)];
    var val3 = data1_65536[(alu12+256)];
    var val4 = data1_65536[(alu12+257)];
    var val5 = data1_65536[(alu12+258)];
    var val6 = data1_65536[(alu12+259)];
    var val7 = data1_65536[(alu12+3)];
    var val8 = data1_65536[(alu12+512)];
    var val9 = data1_65536[(alu12+513)];
    var val10 = data1_65536[(alu12+514)];
    var val11 = data1_65536[(alu12+515)];
    var val12 = data1_65536[(alu12+768)];
    var val13 = data1_65536[(alu12+769)];
    var val14 = data1_65536[(alu12+770)];
    var val15 = data1_65536[(alu12+771)];
    var alu13 = ((gidx0*12288)+(lidx1*768)+cast0);
    var val16 = data2_196608[(alu13+1)];
    var val17 = data2_196608[(alu13+2)];
    var val18 = data2_196608[(alu13+256)];
    var val19 = data2_196608[(alu13+257)];
    var val20 = data2_196608[(alu13+258)];
    var val21 = data2_196608[(alu13+259)];
    var val22 = data2_196608[(alu13+3)];
    var val23 = data2_196608[(alu13+512)];
    var val24 = data2_196608[(alu13+513)];
    var val25 = data2_196608[(alu13+514)];
    var val26 = data2_196608[(alu13+515)];
    var val27 = data2_196608[alu13];
    buf0[0] = (buf0[0]+(val0*val27)+(val1*val16)+(val2*val17)+(val7*val22));
    buf0[1] = (buf0[1]+(val0*val18)+(val1*val19)+(val2*val20)+(val7*val21));
    buf0[10] = (buf0[10]+(val12*val18)+(val13*val19)+(val14*val20)+(val15*val21));
    buf0[11] = (buf0[11]+(val12*val23)+(val13*val24)+(val14*val25)+(val15*val26));
    buf0[2] = (buf0[2]+(val0*val23)+(val1*val24)+(val2*val25)+(val7*val26));
    buf0[3] = (buf0[3]+(val3*val27)+(val4*val16)+(val5*val17)+(val6*val22));
    buf0[4] = (buf0[4]+(val3*val18)+(val4*val19)+(val5*val20)+(val6*val21));
    buf0[5] = (buf0[5]+(val3*val23)+(val4*val24)+(val5*val25)+(val6*val26));
    buf0[6] = (buf0[6]+(val8*val27)+(val9*val16)+(val10*val17)+(val11*val22));
    buf0[7] = (buf0[7]+(val8*val18)+(val9*val19)+(val10*val20)+(val11*val21));
    buf0[8] = (buf0[8]+(val8*val23)+(val9*val24)+(val10*val25)+(val11*val26));
    buf0[9] = (buf0[9]+(val12*val27)+(val13*val16)+(val14*val17)+(val15*val22));
  }
  var alu27 = ((gidx0*48)+(lidx1*3));
  var val28 = data3_768[(alu27+1)];
  var val29 = data3_768[(alu27+2)];
  var val30 = data3_768[alu27];
  var alu28 = (alu27+(gidx1*24576)+(lidx0*3072));
  data0_196608[(alu28+1)] = (buf0[1]+val28);
  data0_196608[(alu28+1536)] = (buf0[6]+val30);
  data0_196608[(alu28+1537)] = (buf0[7]+val28);
  data0_196608[(alu28+1538)] = (buf0[8]+val29);
  data0_196608[(alu28+2)] = (buf0[2]+val29);
  data0_196608[(alu28+2304)] = (buf0[9]+val30);
  data0_196608[(alu28+2305)] = (buf0[10]+val28);
  data0_196608[(alu28+2306)] = (buf0[11]+val29);
  data0_196608[(alu28+768)] = (buf0[3]+val30);
  data0_196608[(alu28+769)] = (buf0[4]+val28);
  data0_196608[(alu28+770)] = (buf0[5]+val29);
  data0_196608[alu28] = (buf0[0]+val30);
}`;

const r_8_16_8_4_4_4_dispatch_16 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[alu4];
    var val1 = data1_196608[(alu4+3072)];
    var val2 = data1_196608[(alu4+6144)];
    var val3 = data1_196608[(alu4+9216)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const r_8_16_8_4_4_4_dispatch_17 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[(alu4+256)];
    var val1 = data1_196608[(alu4+3328)];
    var val2 = data1_196608[(alu4+6400)];
    var val3 = data1_196608[(alu4+9472)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const E_2_32_8_16_2_4_dispatch_18 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+1)];
  var val1 = data1_196608[(alu0+32)];
  var val2 = data1_196608[(alu0+33)];
  var val3 = data1_196608[(alu0+64)];
  var val4 = data1_196608[(alu0+65)];
  var val5 = data1_196608[(alu0+96)];
  var val6 = data1_196608[(alu0+97)];
  var val7 = data1_196608[alu0];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val13);
  var alu10 = (val7*alu5*val12);
  var alu11 = (val1*alu6*val12);
  var alu12 = (val2*alu6*val13);
  var alu13 = (val3*alu7*val12);
  var alu14 = (val4*alu7*val13);
  var alu15 = (val5*alu8*val12);
  var alu16 = (val6*alu8*val13);
  data0_65536[alu4] = ((alu10*val14)-(alu9*val16));
  data0_65536[(alu4+1)] = ((alu9*val15)+(alu10*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const E_2_32_8_16_2_4_dispatch_19 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+256)];
  var val1 = data1_196608[(alu0+257)];
  var val2 = data1_196608[(alu0+288)];
  var val3 = data1_196608[(alu0+289)];
  var val4 = data1_196608[(alu0+320)];
  var val5 = data1_196608[(alu0+321)];
  var val6 = data1_196608[(alu0+352)];
  var val7 = data1_196608[(alu0+353)];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val12);
  var alu10 = (val1*alu5*val13);
  var alu11 = (val2*alu6*val12);
  var alu12 = (val3*alu6*val13);
  var alu13 = (val4*alu7*val12);
  var alu14 = (val5*alu7*val13);
  var alu15 = (val6*alu8*val12);
  var alu16 = (val7*alu8*val13);
  data0_65536[alu4] = ((alu9*val14)-(alu10*val16));
  data0_65536[(alu4+1)] = ((alu10*val15)+(alu9*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const r_8_8_4_8_16_4_4_32_dispatch_20 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_524288:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
  var cast2 = bitcast<i32>((cast1<<13u));
  var cast3 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<10u))+bitcast<i32>((cast3<<7u))+cast2);
  var val0 = data1_65536[alu0];
  var val1 = data1_65536[(alu0+1)];
  var val2 = data1_65536[(alu0+10)];
  var val3 = data1_65536[(alu0+100)];
  var val4 = data1_65536[(alu0+101)];
  var val5 = data1_65536[(alu0+102)];
  var val6 = data1_65536[(alu0+103)];
  var val7 = data1_65536[(alu0+104)];
  var val8 = data1_65536[(alu0+105)];
  var val9 = data1_65536[(alu0+106)];
  var val10 = data1_65536[(alu0+107)];
  var val11 = data1_65536[(alu0+108)];
  var val12 = data1_65536[(alu0+109)];
  var val13 = data1_65536[(alu0+11)];
  var val14 = data1_65536[(alu0+110)];
  var val15 = data1_65536[(alu0+111)];
  var val16 = data1_65536[(alu0+112)];
  var val17 = data1_65536[(alu0+113)];
  var val18 = data1_65536[(alu0+114)];
  var val19 = data1_65536[(alu0+115)];
  var val20 = data1_65536[(alu0+116)];
  var val21 = data1_65536[(alu0+117)];
  var val22 = data1_65536[(alu0+118)];
  var val23 = data1_65536[(alu0+119)];
  var val24 = data1_65536[(alu0+12)];
  var val25 = data1_65536[(alu0+120)];
  var val26 = data1_65536[(alu0+121)];
  var val27 = data1_65536[(alu0+122)];
  var val28 = data1_65536[(alu0+123)];
  var val29 = data1_65536[(alu0+124)];
  var val30 = data1_65536[(alu0+125)];
  var val31 = data1_65536[(alu0+126)];
  var val32 = data1_65536[(alu0+127)];
  var val33 = data1_65536[(alu0+13)];
  var val34 = data1_65536[(alu0+14)];
  var val35 = data1_65536[(alu0+15)];
  var val36 = data1_65536[(alu0+16)];
  var val37 = data1_65536[(alu0+17)];
  var val38 = data1_65536[(alu0+18)];
  var val39 = data1_65536[(alu0+19)];
  var val40 = data1_65536[(alu0+2)];
  var val41 = data1_65536[(alu0+20)];
  var val42 = data1_65536[(alu0+21)];
  var val43 = data1_65536[(alu0+22)];
  var val44 = data1_65536[(alu0+23)];
  var val45 = data1_65536[(alu0+24)];
  var val46 = data1_65536[(alu0+25)];
  var val47 = data1_65536[(alu0+26)];
  var val48 = data1_65536[(alu0+27)];
  var val49 = data1_65536[(alu0+28)];
  var val50 = data1_65536[(alu0+29)];
  var val51 = data1_65536[(alu0+3)];
  var val52 = data1_65536[(alu0+30)];
  var val53 = data1_65536[(alu0+31)];
  var val54 = data1_65536[(alu0+32)];
  var val55 = data1_65536[(alu0+33)];
  var val56 = data1_65536[(alu0+34)];
  var val57 = data1_65536[(alu0+35)];
  var val58 = data1_65536[(alu0+36)];
  var val59 = data1_65536[(alu0+37)];
  var val60 = data1_65536[(alu0+38)];
  var val61 = data1_65536[(alu0+39)];
  var val62 = data1_65536[(alu0+4)];
  var val63 = data1_65536[(alu0+40)];
  var val64 = data1_65536[(alu0+41)];
  var val65 = data1_65536[(alu0+42)];
  var val66 = data1_65536[(alu0+43)];
  var val67 = data1_65536[(alu0+44)];
  var val68 = data1_65536[(alu0+45)];
  var val69 = data1_65536[(alu0+46)];
  var val70 = data1_65536[(alu0+47)];
  var val71 = data1_65536[(alu0+48)];
  var val72 = data1_65536[(alu0+49)];
  var val73 = data1_65536[(alu0+5)];
  var val74 = data1_65536[(alu0+50)];
  var val75 = data1_65536[(alu0+51)];
  var val76 = data1_65536[(alu0+52)];
  var val77 = data1_65536[(alu0+53)];
  var val78 = data1_65536[(alu0+54)];
  var val79 = data1_65536[(alu0+55)];
  var val80 = data1_65536[(alu0+56)];
  var val81 = data1_65536[(alu0+57)];
  var val82 = data1_65536[(alu0+58)];
  var val83 = data1_65536[(alu0+59)];
  var val84 = data1_65536[(alu0+6)];
  var val85 = data1_65536[(alu0+60)];
  var val86 = data1_65536[(alu0+61)];
  var val87 = data1_65536[(alu0+62)];
  var val88 = data1_65536[(alu0+63)];
  var val89 = data1_65536[(alu0+64)];
  var val90 = data1_65536[(alu0+65)];
  var val91 = data1_65536[(alu0+66)];
  var val92 = data1_65536[(alu0+67)];
  var val93 = data1_65536[(alu0+68)];
  var val94 = data1_65536[(alu0+69)];
  var val95 = data1_65536[(alu0+7)];
  var val96 = data1_65536[(alu0+70)];
  var val97 = data1_65536[(alu0+71)];
  var val98 = data1_65536[(alu0+72)];
  var val99 = data1_65536[(alu0+73)];
  var val100 = data1_65536[(alu0+74)];
  var val101 = data1_65536[(alu0+75)];
  var val102 = data1_65536[(alu0+76)];
  var val103 = data1_65536[(alu0+77)];
  var val104 = data1_65536[(alu0+78)];
  var val105 = data1_65536[(alu0+79)];
  var val106 = data1_65536[(alu0+8)];
  var val107 = data1_65536[(alu0+80)];
  var val108 = data1_65536[(alu0+81)];
  var val109 = data1_65536[(alu0+82)];
  var val110 = data1_65536[(alu0+83)];
  var val111 = data1_65536[(alu0+84)];
  var val112 = data1_65536[(alu0+85)];
  var val113 = data1_65536[(alu0+86)];
  var val114 = data1_65536[(alu0+87)];
  var val115 = data1_65536[(alu0+88)];
  var val116 = data1_65536[(alu0+89)];
  var val117 = data1_65536[(alu0+9)];
  var val118 = data1_65536[(alu0+90)];
  var val119 = data1_65536[(alu0+91)];
  var val120 = data1_65536[(alu0+92)];
  var val121 = data1_65536[(alu0+93)];
  var val122 = data1_65536[(alu0+94)];
  var val123 = data1_65536[(alu0+95)];
  var val124 = data1_65536[(alu0+96)];
  var val125 = data1_65536[(alu0+97)];
  var val126 = data1_65536[(alu0+98)];
  var val127 = data1_65536[(alu0+99)];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast4 = bitcast<u32>(gidx0);
  var cast5 = bitcast<u32>(lidx1);
  var alu1 = (bitcast<i32>((cast4<<11u))+bitcast<i32>((cast5<<7u))+cast2);
  var val128 = data2_65536[alu1];
  var val129 = data2_65536[(alu1+1)];
  var val130 = data2_65536[(alu1+10)];
  var val131 = data2_65536[(alu1+100)];
  var val132 = data2_65536[(alu1+101)];
  var val133 = data2_65536[(alu1+102)];
  var val134 = data2_65536[(alu1+103)];
  var val135 = data2_65536[(alu1+104)];
  var val136 = data2_65536[(alu1+105)];
  var val137 = data2_65536[(alu1+106)];
  var val138 = data2_65536[(alu1+107)];
  var val139 = data2_65536[(alu1+108)];
  var val140 = data2_65536[(alu1+109)];
  var val141 = data2_65536[(alu1+11)];
  var val142 = data2_65536[(alu1+110)];
  var val143 = data2_65536[(alu1+111)];
  var val144 = data2_65536[(alu1+112)];
  var val145 = data2_65536[(alu1+113)];
  var val146 = data2_65536[(alu1+114)];
  var val147 = data2_65536[(alu1+115)];
  var val148 = data2_65536[(alu1+116)];
  var val149 = data2_65536[(alu1+117)];
  var val150 = data2_65536[(alu1+118)];
  var val151 = data2_65536[(alu1+119)];
  var val152 = data2_65536[(alu1+12)];
  var val153 = data2_65536[(alu1+120)];
  var val154 = data2_65536[(alu1+121)];
  var val155 = data2_65536[(alu1+122)];
  var val156 = data2_65536[(alu1+123)];
  var val157 = data2_65536[(alu1+124)];
  var val158 = data2_65536[(alu1+125)];
  var val159 = data2_65536[(alu1+126)];
  var val160 = data2_65536[(alu1+127)];
  var val161 = data2_65536[(alu1+13)];
  var val162 = data2_65536[(alu1+14)];
  var val163 = data2_65536[(alu1+15)];
  var val164 = data2_65536[(alu1+16)];
  var val165 = data2_65536[(alu1+17)];
  var val166 = data2_65536[(alu1+18)];
  var val167 = data2_65536[(alu1+19)];
  var val168 = data2_65536[(alu1+2)];
  var val169 = data2_65536[(alu1+20)];
  var val170 = data2_65536[(alu1+21)];
  var val171 = data2_65536[(alu1+22)];
  var val172 = data2_65536[(alu1+23)];
  var val173 = data2_65536[(alu1+24)];
  var val174 = data2_65536[(alu1+25)];
  var val175 = data2_65536[(alu1+26)];
  var val176 = data2_65536[(alu1+27)];
  var val177 = data2_65536[(alu1+28)];
  var val178 = data2_65536[(alu1+29)];
  var val179 = data2_65536[(alu1+3)];
  var val180 = data2_65536[(alu1+30)];
  var val181 = data2_65536[(alu1+31)];
  var val182 = data2_65536[(alu1+32)];
  var val183 = data2_65536[(alu1+33)];
  var val184 = data2_65536[(alu1+34)];
  var val185 = data2_65536[(alu1+35)];
  var val186 = data2_65536[(alu1+36)];
  var val187 = data2_65536[(alu1+37)];
  var val188 = data2_65536[(alu1+38)];
  var val189 = data2_65536[(alu1+39)];
  var val190 = data2_65536[(alu1+4)];
  var val191 = data2_65536[(alu1+40)];
  var val192 = data2_65536[(alu1+41)];
  var val193 = data2_65536[(alu1+42)];
  var val194 = data2_65536[(alu1+43)];
  var val195 = data2_65536[(alu1+44)];
  var val196 = data2_65536[(alu1+45)];
  var val197 = data2_65536[(alu1+46)];
  var val198 = data2_65536[(alu1+47)];
  var val199 = data2_65536[(alu1+48)];
  var val200 = data2_65536[(alu1+49)];
  var val201 = data2_65536[(alu1+5)];
  var val202 = data2_65536[(alu1+50)];
  var val203 = data2_65536[(alu1+51)];
  var val204 = data2_65536[(alu1+52)];
  var val205 = data2_65536[(alu1+53)];
  var val206 = data2_65536[(alu1+54)];
  var val207 = data2_65536[(alu1+55)];
  var val208 = data2_65536[(alu1+56)];
  var val209 = data2_65536[(alu1+57)];
  var val210 = data2_65536[(alu1+58)];
  var val211 = data2_65536[(alu1+59)];
  var val212 = data2_65536[(alu1+6)];
  var val213 = data2_65536[(alu1+60)];
  var val214 = data2_65536[(alu1+61)];
  var val215 = data2_65536[(alu1+62)];
  var val216 = data2_65536[(alu1+63)];
  var val217 = data2_65536[(alu1+64)];
  var val218 = data2_65536[(alu1+65)];
  var val219 = data2_65536[(alu1+66)];
  var val220 = data2_65536[(alu1+67)];
  var val221 = data2_65536[(alu1+68)];
  var val222 = data2_65536[(alu1+69)];
  var val223 = data2_65536[(alu1+7)];
  var val224 = data2_65536[(alu1+70)];
  var val225 = data2_65536[(alu1+71)];
  var val226 = data2_65536[(alu1+72)];
  var val227 = data2_65536[(alu1+73)];
  var val228 = data2_65536[(alu1+74)];
  var val229 = data2_65536[(alu1+75)];
  var val230 = data2_65536[(alu1+76)];
  var val231 = data2_65536[(alu1+77)];
  var val232 = data2_65536[(alu1+78)];
  var val233 = data2_65536[(alu1+79)];
  var val234 = data2_65536[(alu1+8)];
  var val235 = data2_65536[(alu1+80)];
  var val236 = data2_65536[(alu1+81)];
  var val237 = data2_65536[(alu1+82)];
  var val238 = data2_65536[(alu1+83)];
  var val239 = data2_65536[(alu1+84)];
  var val240 = data2_65536[(alu1+85)];
  var val241 = data2_65536[(alu1+86)];
  var val242 = data2_65536[(alu1+87)];
  var val243 = data2_65536[(alu1+88)];
  var val244 = data2_65536[(alu1+89)];
  var val245 = data2_65536[(alu1+9)];
  var val246 = data2_65536[(alu1+90)];
  var val247 = data2_65536[(alu1+91)];
  var val248 = data2_65536[(alu1+92)];
  var val249 = data2_65536[(alu1+93)];
  var val250 = data2_65536[(alu1+94)];
  var val251 = data2_65536[(alu1+95)];
  var val252 = data2_65536[(alu1+96)];
  var val253 = data2_65536[(alu1+97)];
  var val254 = data2_65536[(alu1+98)];
  var val255 = data2_65536[(alu1+99)];
  var alu2 = (bitcast<i32>((cast4<<6u))+bitcast<i32>((cast5<<2u))+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast1<<16u)));
  data0_524288[alu2] = (((val0*val128)+(val1*val129)+(val40*val168)+(val51*val179)+(val62*val190)+(val73*val201)+(val84*val212)+(val95*val223)+(val106*val234)+(val117*val245)+(val2*val130)+(val13*val141)+(val24*val152)+(val33*val161)+(val34*val162)+(val35*val163)+(val36*val164)+(val37*val165)+(val38*val166)+(val39*val167)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val173)+(val46*val174)+(val47*val175)+(val48*val176)+(val49*val177)+(val50*val178)+(val52*val180)+(val53*val181))*0.1767766922712326f);
  data0_524288[(alu2+1)] = (((val0*val182)+(val1*val183)+(val40*val184)+(val51*val185)+(val62*val186)+(val73*val187)+(val84*val188)+(val95*val189)+(val106*val191)+(val117*val192)+(val2*val193)+(val13*val194)+(val24*val195)+(val33*val196)+(val34*val197)+(val35*val198)+(val36*val199)+(val37*val200)+(val38*val202)+(val39*val203)+(val41*val204)+(val42*val205)+(val43*val206)+(val44*val207)+(val45*val208)+(val46*val209)+(val47*val210)+(val48*val211)+(val49*val213)+(val50*val214)+(val52*val215)+(val53*val216))*0.1767766922712326f);
  data0_524288[(alu2+2)] = (((val0*val217)+(val1*val218)+(val40*val219)+(val51*val220)+(val62*val221)+(val73*val222)+(val84*val224)+(val95*val225)+(val106*val226)+(val117*val227)+(val2*val228)+(val13*val229)+(val24*val230)+(val33*val231)+(val34*val232)+(val35*val233)+(val36*val235)+(val37*val236)+(val38*val237)+(val39*val238)+(val41*val239)+(val42*val240)+(val43*val241)+(val44*val242)+(val45*val243)+(val46*val244)+(val47*val246)+(val48*val247)+(val49*val248)+(val50*val249)+(val52*val250)+(val53*val251))*0.1767766922712326f);
  data0_524288[(alu2+256)] = (((val54*val128)+(val55*val129)+(val56*val168)+(val57*val179)+(val58*val190)+(val59*val201)+(val60*val212)+(val61*val223)+(val63*val234)+(val64*val245)+(val65*val130)+(val66*val141)+(val67*val152)+(val68*val161)+(val69*val162)+(val70*val163)+(val71*val164)+(val72*val165)+(val74*val166)+(val75*val167)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val173)+(val81*val174)+(val82*val175)+(val83*val176)+(val85*val177)+(val86*val178)+(val87*val180)+(val88*val181))*0.1767766922712326f);
  data0_524288[(alu2+257)] = (((val54*val182)+(val55*val183)+(val56*val184)+(val57*val185)+(val58*val186)+(val59*val187)+(val60*val188)+(val61*val189)+(val63*val191)+(val64*val192)+(val65*val193)+(val66*val194)+(val67*val195)+(val68*val196)+(val69*val197)+(val70*val198)+(val71*val199)+(val72*val200)+(val74*val202)+(val75*val203)+(val76*val204)+(val77*val205)+(val78*val206)+(val79*val207)+(val80*val208)+(val81*val209)+(val82*val210)+(val83*val211)+(val85*val213)+(val86*val214)+(val87*val215)+(val88*val216))*0.1767766922712326f);
  data0_524288[(alu2+258)] = (((val54*val217)+(val55*val218)+(val56*val219)+(val57*val220)+(val58*val221)+(val59*val222)+(val60*val224)+(val61*val225)+(val63*val226)+(val64*val227)+(val65*val228)+(val66*val229)+(val67*val230)+(val68*val231)+(val69*val232)+(val70*val233)+(val71*val235)+(val72*val236)+(val74*val237)+(val75*val238)+(val76*val239)+(val77*val240)+(val78*val241)+(val79*val242)+(val80*val243)+(val81*val244)+(val82*val246)+(val83*val247)+(val85*val248)+(val86*val249)+(val87*val250)+(val88*val251))*0.1767766922712326f);
  data0_524288[(alu2+259)] = (((val54*val252)+(val55*val253)+(val56*val254)+(val57*val255)+(val58*val131)+(val59*val132)+(val60*val133)+(val61*val134)+(val63*val135)+(val64*val136)+(val65*val137)+(val66*val138)+(val67*val139)+(val68*val140)+(val69*val142)+(val70*val143)+(val71*val144)+(val72*val145)+(val74*val146)+(val75*val147)+(val76*val148)+(val77*val149)+(val78*val150)+(val79*val151)+(val80*val153)+(val81*val154)+(val82*val155)+(val83*val156)+(val85*val157)+(val86*val158)+(val87*val159)+(val88*val160))*0.1767766922712326f);
  data0_524288[(alu2+3)] = (((val0*val252)+(val1*val253)+(val40*val254)+(val51*val255)+(val62*val131)+(val73*val132)+(val84*val133)+(val95*val134)+(val106*val135)+(val117*val136)+(val2*val137)+(val13*val138)+(val24*val139)+(val33*val140)+(val34*val142)+(val35*val143)+(val36*val144)+(val37*val145)+(val38*val146)+(val39*val147)+(val41*val148)+(val42*val149)+(val43*val150)+(val44*val151)+(val45*val153)+(val46*val154)+(val47*val155)+(val48*val156)+(val49*val157)+(val50*val158)+(val52*val159)+(val53*val160))*0.1767766922712326f);
  data0_524288[(alu2+512)] = (((val89*val128)+(val90*val129)+(val91*val168)+(val92*val179)+(val93*val190)+(val94*val201)+(val96*val212)+(val97*val223)+(val98*val234)+(val99*val245)+(val100*val130)+(val101*val141)+(val102*val152)+(val103*val161)+(val104*val162)+(val105*val163)+(val107*val164)+(val108*val165)+(val109*val166)+(val110*val167)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val173)+(val116*val174)+(val118*val175)+(val119*val176)+(val120*val177)+(val121*val178)+(val122*val180)+(val123*val181))*0.1767766922712326f);
  data0_524288[(alu2+513)] = (((val89*val182)+(val90*val183)+(val91*val184)+(val92*val185)+(val93*val186)+(val94*val187)+(val96*val188)+(val97*val189)+(val98*val191)+(val99*val192)+(val100*val193)+(val101*val194)+(val102*val195)+(val103*val196)+(val104*val197)+(val105*val198)+(val107*val199)+(val108*val200)+(val109*val202)+(val110*val203)+(val111*val204)+(val112*val205)+(val113*val206)+(val114*val207)+(val115*val208)+(val116*val209)+(val118*val210)+(val119*val211)+(val120*val213)+(val121*val214)+(val122*val215)+(val123*val216))*0.1767766922712326f);
  data0_524288[(alu2+514)] = (((val89*val217)+(val90*val218)+(val91*val219)+(val92*val220)+(val93*val221)+(val94*val222)+(val96*val224)+(val97*val225)+(val98*val226)+(val99*val227)+(val100*val228)+(val101*val229)+(val102*val230)+(val103*val231)+(val104*val232)+(val105*val233)+(val107*val235)+(val108*val236)+(val109*val237)+(val110*val238)+(val111*val239)+(val112*val240)+(val113*val241)+(val114*val242)+(val115*val243)+(val116*val244)+(val118*val246)+(val119*val247)+(val120*val248)+(val121*val249)+(val122*val250)+(val123*val251))*0.1767766922712326f);
  data0_524288[(alu2+515)] = (((val89*val252)+(val90*val253)+(val91*val254)+(val92*val255)+(val93*val131)+(val94*val132)+(val96*val133)+(val97*val134)+(val98*val135)+(val99*val136)+(val100*val137)+(val101*val138)+(val102*val139)+(val103*val140)+(val104*val142)+(val105*val143)+(val107*val144)+(val108*val145)+(val109*val146)+(val110*val147)+(val111*val148)+(val112*val149)+(val113*val150)+(val114*val151)+(val115*val153)+(val116*val154)+(val118*val155)+(val119*val156)+(val120*val157)+(val121*val158)+(val122*val159)+(val123*val160))*0.1767766922712326f);
  data0_524288[(alu2+768)] = (((val124*val128)+(val125*val129)+(val126*val168)+(val127*val179)+(val3*val190)+(val4*val201)+(val5*val212)+(val6*val223)+(val7*val234)+(val8*val245)+(val9*val130)+(val10*val141)+(val11*val152)+(val12*val161)+(val14*val162)+(val15*val163)+(val16*val164)+(val17*val165)+(val18*val166)+(val19*val167)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val173)+(val26*val174)+(val27*val175)+(val28*val176)+(val29*val177)+(val30*val178)+(val31*val180)+(val32*val181))*0.1767766922712326f);
  data0_524288[(alu2+769)] = (((val124*val182)+(val125*val183)+(val126*val184)+(val127*val185)+(val3*val186)+(val4*val187)+(val5*val188)+(val6*val189)+(val7*val191)+(val8*val192)+(val9*val193)+(val10*val194)+(val11*val195)+(val12*val196)+(val14*val197)+(val15*val198)+(val16*val199)+(val17*val200)+(val18*val202)+(val19*val203)+(val20*val204)+(val21*val205)+(val22*val206)+(val23*val207)+(val25*val208)+(val26*val209)+(val27*val210)+(val28*val211)+(val29*val213)+(val30*val214)+(val31*val215)+(val32*val216))*0.1767766922712326f);
  data0_524288[(alu2+770)] = (((val124*val217)+(val125*val218)+(val126*val219)+(val127*val220)+(val3*val221)+(val4*val222)+(val5*val224)+(val6*val225)+(val7*val226)+(val8*val227)+(val9*val228)+(val10*val229)+(val11*val230)+(val12*val231)+(val14*val232)+(val15*val233)+(val16*val235)+(val17*val236)+(val18*val237)+(val19*val238)+(val20*val239)+(val21*val240)+(val22*val241)+(val23*val242)+(val25*val243)+(val26*val244)+(val27*val246)+(val28*val247)+(val29*val248)+(val30*val249)+(val31*val250)+(val32*val251))*0.1767766922712326f);
  data0_524288[(alu2+771)] = (((val124*val252)+(val125*val253)+(val126*val254)+(val127*val255)+(val3*val131)+(val4*val132)+(val5*val133)+(val6*val134)+(val7*val135)+(val8*val136)+(val9*val137)+(val10*val138)+(val11*val139)+(val12*val140)+(val14*val142)+(val15*val143)+(val16*val144)+(val17*val145)+(val18*val146)+(val19*val147)+(val20*val148)+(val21*val149)+(val22*val150)+(val23*val151)+(val25*val153)+(val26*val154)+(val27*val155)+(val28*val156)+(val29*val157)+(val30*val158)+(val31*val159)+(val32*val160))*0.1767766922712326f);
}`;

const r_2048_16_16_dispatch_21 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = (f32(-INFINITY));
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val0 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    var val1 = buf0[0];
    var alu1 = select(val1,val0,(val1<val0));
    buf0[0] = alu1;
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = (f32(-INFINITY));
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf1[0];
    var val3 = buf2[Lidx102];
    var alu7 = select(val2,val3,(val2<val3));
    buf1[0] = alu7;
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_2048_16_16_dispatch_22 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var val0 = data2_2048[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val1 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    buf0[0] = (buf0[0]+exp2(((val1-val0)*1.4426950216293335f)));
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf2[Lidx102];
    buf1[0] = (buf1[0]+val2);
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_8_4_16_8_4_4_64_4_dispatch_23 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2048:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_196608:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u))+bitcast<i32>((cast1<<8u)));
  var val0 = data2_2048[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_2048[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_2048[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_2048[alu3];
  var lidx1 = i32(lindex.y); /* 8 */
  var cast3 = bitcast<i32>((bitcast<u32>(lidx1)<<2u));
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var alu20 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast2<<10u))+bitcast<i32>((bitcast<u32>(Lidx0)<<2u))+bitcast<i32>((cast1<<16u)));
    var val4 = data1_524288[alu20];
    var val5 = data1_524288[(alu20+1)];
    var val6 = data1_524288[(alu20+2)];
    var val7 = data1_524288[(alu20+256)];
    var val8 = data1_524288[(alu20+257)];
    var val9 = data1_524288[(alu20+258)];
    var val10 = data1_524288[(alu20+259)];
    var val11 = data1_524288[(alu20+3)];
    var val12 = data1_524288[(alu20+512)];
    var val13 = data1_524288[(alu20+513)];
    var val14 = data1_524288[(alu20+514)];
    var val15 = data1_524288[(alu20+515)];
    var val16 = data1_524288[(alu20+768)];
    var val17 = data1_524288[(alu20+769)];
    var val18 = data1_524288[(alu20+770)];
    var val19 = data1_524288[(alu20+771)];
    var alu21 = (bitcast<i32>((cast1<<5u))+cast3+(Lidx0*3072));
    var val20 = data4_196608[(alu21+1280)];
    var val21 = data4_196608[(alu21+1281)];
    var val22 = data4_196608[(alu21+1282)];
    var val23 = data4_196608[(alu21+1283)];
    var val24 = data4_196608[(alu21+2048)];
    var val25 = data4_196608[(alu21+2049)];
    var val26 = data4_196608[(alu21+2050)];
    var val27 = data4_196608[(alu21+2051)];
    var val28 = data4_196608[(alu21+2816)];
    var val29 = data4_196608[(alu21+2817)];
    var val30 = data4_196608[(alu21+2818)];
    var val31 = data4_196608[(alu21+2819)];
    var val32 = data4_196608[(alu21+512)];
    var val33 = data4_196608[(alu21+513)];
    var val34 = data4_196608[(alu21+514)];
    var val35 = data4_196608[(alu21+515)];
    var alu22 = exp2(((val4-val0)*1.4426950216293335f));
    var alu23 = exp2(((val5-val0)*1.4426950216293335f));
    var alu24 = exp2(((val6-val0)*1.4426950216293335f));
    var alu25 = exp2(((val7-val1)*1.4426950216293335f));
    var alu26 = exp2(((val8-val1)*1.4426950216293335f));
    var alu27 = exp2(((val9-val1)*1.4426950216293335f));
    var alu28 = exp2(((val10-val1)*1.4426950216293335f));
    var alu29 = exp2(((val11-val0)*1.4426950216293335f));
    var alu30 = exp2(((val12-val2)*1.4426950216293335f));
    var alu31 = exp2(((val13-val2)*1.4426950216293335f));
    var alu32 = exp2(((val14-val2)*1.4426950216293335f));
    var alu33 = exp2(((val15-val2)*1.4426950216293335f));
    var alu34 = exp2(((val16-val3)*1.4426950216293335f));
    var alu35 = exp2(((val17-val3)*1.4426950216293335f));
    var alu36 = exp2(((val18-val3)*1.4426950216293335f));
    var alu37 = exp2(((val19-val3)*1.4426950216293335f));
    buf0[0] = (buf0[0]+(alu22*val32)+(alu23*val20)+(alu24*val24)+(alu29*val28));
    buf0[1] = (buf0[1]+(alu22*val33)+(alu23*val21)+(alu24*val25)+(alu29*val29));
    buf0[10] = (buf0[10]+(alu30*val34)+(alu31*val22)+(alu32*val26)+(alu33*val30));
    buf0[11] = (buf0[11]+(alu30*val35)+(alu31*val23)+(alu32*val27)+(alu33*val31));
    buf0[12] = (buf0[12]+(alu34*val32)+(alu35*val20)+(alu36*val24)+(alu37*val28));
    buf0[13] = (buf0[13]+(alu34*val33)+(alu35*val21)+(alu36*val25)+(alu37*val29));
    buf0[14] = (buf0[14]+(alu34*val34)+(alu35*val22)+(alu36*val26)+(alu37*val30));
    buf0[15] = (buf0[15]+(alu34*val35)+(alu35*val23)+(alu36*val27)+(alu37*val31));
    buf0[2] = (buf0[2]+(alu22*val34)+(alu23*val22)+(alu24*val26)+(alu29*val30));
    buf0[3] = (buf0[3]+(alu22*val35)+(alu23*val23)+(alu24*val27)+(alu29*val31));
    buf0[4] = (buf0[4]+(alu25*val32)+(alu26*val20)+(alu27*val24)+(alu28*val28));
    buf0[5] = (buf0[5]+(alu25*val33)+(alu26*val21)+(alu27*val25)+(alu28*val29));
    buf0[6] = (buf0[6]+(alu25*val34)+(alu26*val22)+(alu27*val26)+(alu28*val30));
    buf0[7] = (buf0[7]+(alu25*val35)+(alu26*val23)+(alu27*val27)+(alu28*val31));
    buf0[8] = (buf0[8]+(alu30*val32)+(alu31*val20)+(alu32*val24)+(alu33*val28));
    buf0[9] = (buf0[9]+(alu30*val33)+(alu31*val21)+(alu32*val25)+(alu33*val29));
  }
  var val36 = data3_2048[alu0];
  var val37 = data3_2048[alu1];
  var val38 = data3_2048[alu2];
  var val39 = data3_2048[alu3];
  var alu55 = (bitcast<i32>((cast0<<11u))+bitcast<i32>((cast2<<7u))+cast3+bitcast<i32>((cast1<<13u)));
  var alu56 = (1/val36);
  var alu57 = (1/val37);
  var alu58 = (1/val38);
  var alu59 = (1/val39);
  data0_65536[alu55] = (buf0[0]*alu56);
  data0_65536[(alu55+1)] = (buf0[1]*alu56);
  data0_65536[(alu55+2)] = (buf0[2]*alu56);
  data0_65536[(alu55+3)] = (buf0[3]*alu56);
  data0_65536[(alu55+32)] = (buf0[4]*alu57);
  data0_65536[(alu55+33)] = (buf0[5]*alu57);
  data0_65536[(alu55+34)] = (buf0[6]*alu57);
  data0_65536[(alu55+35)] = (buf0[7]*alu57);
  data0_65536[(alu55+64)] = (buf0[8]*alu58);
  data0_65536[(alu55+65)] = (buf0[9]*alu58);
  data0_65536[(alu55+66)] = (buf0[10]*alu58);
  data0_65536[(alu55+67)] = (buf0[11]*alu58);
  data0_65536[(alu55+96)] = (buf0[12]*alu59);
  data0_65536[(alu55+97)] = (buf0[13]*alu59);
  data0_65536[(alu55+98)] = (buf0[14]*alu59);
  data0_65536[(alu55+99)] = (buf0[15]*alu59);
}`;

const r_8_4_8_16_4_4_8_32_dispatch_24 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_65536:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0_0 = 0; Lidx0_0 < 8; Lidx0_0++) {
    var cast4 = bitcast<u32>(Lidx0_0);
    var alu16 = (bitcast<i32>((cast1<<10u))+bitcast<i32>((cast2<<7u))+bitcast<i32>((cast4<<13u)));
    var val0 = data3_65536[alu16];
    var val1 = data3_65536[(alu16+1)];
    var val2 = data3_65536[(alu16+10)];
    var val3 = data3_65536[(alu16+100)];
    var val4 = data3_65536[(alu16+101)];
    var val5 = data3_65536[(alu16+102)];
    var val6 = data3_65536[(alu16+103)];
    var val7 = data3_65536[(alu16+104)];
    var val8 = data3_65536[(alu16+105)];
    var val9 = data3_65536[(alu16+106)];
    var val10 = data3_65536[(alu16+107)];
    var val11 = data3_65536[(alu16+108)];
    var val12 = data3_65536[(alu16+109)];
    var val13 = data3_65536[(alu16+11)];
    var val14 = data3_65536[(alu16+110)];
    var val15 = data3_65536[(alu16+111)];
    var val16 = data3_65536[(alu16+112)];
    var val17 = data3_65536[(alu16+113)];
    var val18 = data3_65536[(alu16+114)];
    var val19 = data3_65536[(alu16+115)];
    var val20 = data3_65536[(alu16+116)];
    var val21 = data3_65536[(alu16+117)];
    var val22 = data3_65536[(alu16+118)];
    var val23 = data3_65536[(alu16+119)];
    var val24 = data3_65536[(alu16+12)];
    var val25 = data3_65536[(alu16+120)];
    var val26 = data3_65536[(alu16+121)];
    var val27 = data3_65536[(alu16+122)];
    var val28 = data3_65536[(alu16+123)];
    var val29 = data3_65536[(alu16+124)];
    var val30 = data3_65536[(alu16+125)];
    var val31 = data3_65536[(alu16+126)];
    var val32 = data3_65536[(alu16+127)];
    var val33 = data3_65536[(alu16+13)];
    var val34 = data3_65536[(alu16+14)];
    var val35 = data3_65536[(alu16+15)];
    var val36 = data3_65536[(alu16+16)];
    var val37 = data3_65536[(alu16+17)];
    var val38 = data3_65536[(alu16+18)];
    var val39 = data3_65536[(alu16+19)];
    var val40 = data3_65536[(alu16+2)];
    var val41 = data3_65536[(alu16+20)];
    var val42 = data3_65536[(alu16+21)];
    var val43 = data3_65536[(alu16+22)];
    var val44 = data3_65536[(alu16+23)];
    var val45 = data3_65536[(alu16+24)];
    var val46 = data3_65536[(alu16+25)];
    var val47 = data3_65536[(alu16+26)];
    var val48 = data3_65536[(alu16+27)];
    var val49 = data3_65536[(alu16+28)];
    var val50 = data3_65536[(alu16+29)];
    var val51 = data3_65536[(alu16+3)];
    var val52 = data3_65536[(alu16+30)];
    var val53 = data3_65536[(alu16+31)];
    var val54 = data3_65536[(alu16+32)];
    var val55 = data3_65536[(alu16+33)];
    var val56 = data3_65536[(alu16+34)];
    var val57 = data3_65536[(alu16+35)];
    var val58 = data3_65536[(alu16+36)];
    var val59 = data3_65536[(alu16+37)];
    var val60 = data3_65536[(alu16+38)];
    var val61 = data3_65536[(alu16+39)];
    var val62 = data3_65536[(alu16+4)];
    var val63 = data3_65536[(alu16+40)];
    var val64 = data3_65536[(alu16+41)];
    var val65 = data3_65536[(alu16+42)];
    var val66 = data3_65536[(alu16+43)];
    var val67 = data3_65536[(alu16+44)];
    var val68 = data3_65536[(alu16+45)];
    var val69 = data3_65536[(alu16+46)];
    var val70 = data3_65536[(alu16+47)];
    var val71 = data3_65536[(alu16+48)];
    var val72 = data3_65536[(alu16+49)];
    var val73 = data3_65536[(alu16+5)];
    var val74 = data3_65536[(alu16+50)];
    var val75 = data3_65536[(alu16+51)];
    var val76 = data3_65536[(alu16+52)];
    var val77 = data3_65536[(alu16+53)];
    var val78 = data3_65536[(alu16+54)];
    var val79 = data3_65536[(alu16+55)];
    var val80 = data3_65536[(alu16+56)];
    var val81 = data3_65536[(alu16+57)];
    var val82 = data3_65536[(alu16+58)];
    var val83 = data3_65536[(alu16+59)];
    var val84 = data3_65536[(alu16+6)];
    var val85 = data3_65536[(alu16+60)];
    var val86 = data3_65536[(alu16+61)];
    var val87 = data3_65536[(alu16+62)];
    var val88 = data3_65536[(alu16+63)];
    var val89 = data3_65536[(alu16+64)];
    var val90 = data3_65536[(alu16+65)];
    var val91 = data3_65536[(alu16+66)];
    var val92 = data3_65536[(alu16+67)];
    var val93 = data3_65536[(alu16+68)];
    var val94 = data3_65536[(alu16+69)];
    var val95 = data3_65536[(alu16+7)];
    var val96 = data3_65536[(alu16+70)];
    var val97 = data3_65536[(alu16+71)];
    var val98 = data3_65536[(alu16+72)];
    var val99 = data3_65536[(alu16+73)];
    var val100 = data3_65536[(alu16+74)];
    var val101 = data3_65536[(alu16+75)];
    var val102 = data3_65536[(alu16+76)];
    var val103 = data3_65536[(alu16+77)];
    var val104 = data3_65536[(alu16+78)];
    var val105 = data3_65536[(alu16+79)];
    var val106 = data3_65536[(alu16+8)];
    var val107 = data3_65536[(alu16+80)];
    var val108 = data3_65536[(alu16+81)];
    var val109 = data3_65536[(alu16+82)];
    var val110 = data3_65536[(alu16+83)];
    var val111 = data3_65536[(alu16+84)];
    var val112 = data3_65536[(alu16+85)];
    var val113 = data3_65536[(alu16+86)];
    var val114 = data3_65536[(alu16+87)];
    var val115 = data3_65536[(alu16+88)];
    var val116 = data3_65536[(alu16+89)];
    var val117 = data3_65536[(alu16+9)];
    var val118 = data3_65536[(alu16+90)];
    var val119 = data3_65536[(alu16+91)];
    var val120 = data3_65536[(alu16+92)];
    var val121 = data3_65536[(alu16+93)];
    var val122 = data3_65536[(alu16+94)];
    var val123 = data3_65536[(alu16+95)];
    var val124 = data3_65536[(alu16+96)];
    var val125 = data3_65536[(alu16+97)];
    var val126 = data3_65536[(alu16+98)];
    var val127 = data3_65536[(alu16+99)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast4<<5u)));
    var val128 = data4_65536[alu17];
    var val129 = data4_65536[(alu17+1)];
    var val130 = data4_65536[(alu17+10)];
    var val131 = data4_65536[(alu17+11)];
    var val132 = data4_65536[(alu17+12)];
    var val133 = data4_65536[(alu17+13)];
    var val134 = data4_65536[(alu17+14)];
    var val135 = data4_65536[(alu17+15)];
    var val136 = data4_65536[(alu17+16)];
    var val137 = data4_65536[(alu17+17)];
    var val138 = data4_65536[(alu17+18)];
    var val139 = data4_65536[(alu17+19)];
    var val140 = data4_65536[(alu17+2)];
    var val141 = data4_65536[(alu17+20)];
    var val142 = data4_65536[(alu17+21)];
    var val143 = data4_65536[(alu17+22)];
    var val144 = data4_65536[(alu17+23)];
    var val145 = data4_65536[(alu17+24)];
    var val146 = data4_65536[(alu17+25)];
    var val147 = data4_65536[(alu17+256)];
    var val148 = data4_65536[(alu17+257)];
    var val149 = data4_65536[(alu17+258)];
    var val150 = data4_65536[(alu17+259)];
    var val151 = data4_65536[(alu17+26)];
    var val152 = data4_65536[(alu17+260)];
    var val153 = data4_65536[(alu17+261)];
    var val154 = data4_65536[(alu17+262)];
    var val155 = data4_65536[(alu17+263)];
    var val156 = data4_65536[(alu17+264)];
    var val157 = data4_65536[(alu17+265)];
    var val158 = data4_65536[(alu17+266)];
    var val159 = data4_65536[(alu17+267)];
    var val160 = data4_65536[(alu17+268)];
    var val161 = data4_65536[(alu17+269)];
    var val162 = data4_65536[(alu17+27)];
    var val163 = data4_65536[(alu17+270)];
    var val164 = data4_65536[(alu17+271)];
    var val165 = data4_65536[(alu17+272)];
    var val166 = data4_65536[(alu17+273)];
    var val167 = data4_65536[(alu17+274)];
    var val168 = data4_65536[(alu17+275)];
    var val169 = data4_65536[(alu17+276)];
    var val170 = data4_65536[(alu17+277)];
    var val171 = data4_65536[(alu17+278)];
    var val172 = data4_65536[(alu17+279)];
    var val173 = data4_65536[(alu17+28)];
    var val174 = data4_65536[(alu17+280)];
    var val175 = data4_65536[(alu17+281)];
    var val176 = data4_65536[(alu17+282)];
    var val177 = data4_65536[(alu17+283)];
    var val178 = data4_65536[(alu17+284)];
    var val179 = data4_65536[(alu17+285)];
    var val180 = data4_65536[(alu17+286)];
    var val181 = data4_65536[(alu17+287)];
    var val182 = data4_65536[(alu17+29)];
    var val183 = data4_65536[(alu17+3)];
    var val184 = data4_65536[(alu17+30)];
    var val185 = data4_65536[(alu17+31)];
    var val186 = data4_65536[(alu17+4)];
    var val187 = data4_65536[(alu17+5)];
    var val188 = data4_65536[(alu17+512)];
    var val189 = data4_65536[(alu17+513)];
    var val190 = data4_65536[(alu17+514)];
    var val191 = data4_65536[(alu17+515)];
    var val192 = data4_65536[(alu17+516)];
    var val193 = data4_65536[(alu17+517)];
    var val194 = data4_65536[(alu17+518)];
    var val195 = data4_65536[(alu17+519)];
    var val196 = data4_65536[(alu17+520)];
    var val197 = data4_65536[(alu17+521)];
    var val198 = data4_65536[(alu17+522)];
    var val199 = data4_65536[(alu17+523)];
    var val200 = data4_65536[(alu17+524)];
    var val201 = data4_65536[(alu17+525)];
    var val202 = data4_65536[(alu17+526)];
    var val203 = data4_65536[(alu17+527)];
    var val204 = data4_65536[(alu17+528)];
    var val205 = data4_65536[(alu17+529)];
    var val206 = data4_65536[(alu17+530)];
    var val207 = data4_65536[(alu17+531)];
    var val208 = data4_65536[(alu17+532)];
    var val209 = data4_65536[(alu17+533)];
    var val210 = data4_65536[(alu17+534)];
    var val211 = data4_65536[(alu17+535)];
    var val212 = data4_65536[(alu17+536)];
    var val213 = data4_65536[(alu17+537)];
    var val214 = data4_65536[(alu17+538)];
    var val215 = data4_65536[(alu17+539)];
    var val216 = data4_65536[(alu17+540)];
    var val217 = data4_65536[(alu17+541)];
    var val218 = data4_65536[(alu17+542)];
    var val219 = data4_65536[(alu17+543)];
    var val220 = data4_65536[(alu17+6)];
    var val221 = data4_65536[(alu17+7)];
    var val222 = data4_65536[(alu17+768)];
    var val223 = data4_65536[(alu17+769)];
    var val224 = data4_65536[(alu17+770)];
    var val225 = data4_65536[(alu17+771)];
    var val226 = data4_65536[(alu17+772)];
    var val227 = data4_65536[(alu17+773)];
    var val228 = data4_65536[(alu17+774)];
    var val229 = data4_65536[(alu17+775)];
    var val230 = data4_65536[(alu17+776)];
    var val231 = data4_65536[(alu17+777)];
    var val232 = data4_65536[(alu17+778)];
    var val233 = data4_65536[(alu17+779)];
    var val234 = data4_65536[(alu17+780)];
    var val235 = data4_65536[(alu17+781)];
    var val236 = data4_65536[(alu17+782)];
    var val237 = data4_65536[(alu17+783)];
    var val238 = data4_65536[(alu17+784)];
    var val239 = data4_65536[(alu17+785)];
    var val240 = data4_65536[(alu17+786)];
    var val241 = data4_65536[(alu17+787)];
    var val242 = data4_65536[(alu17+788)];
    var val243 = data4_65536[(alu17+789)];
    var val244 = data4_65536[(alu17+790)];
    var val245 = data4_65536[(alu17+791)];
    var val246 = data4_65536[(alu17+792)];
    var val247 = data4_65536[(alu17+793)];
    var val248 = data4_65536[(alu17+794)];
    var val249 = data4_65536[(alu17+795)];
    var val250 = data4_65536[(alu17+796)];
    var val251 = data4_65536[(alu17+797)];
    var val252 = data4_65536[(alu17+798)];
    var val253 = data4_65536[(alu17+799)];
    var val254 = data4_65536[(alu17+8)];
    var val255 = data4_65536[(alu17+9)];
    buf0[0] = (buf0[0]+(val0*val128)+(val1*val129)+(val40*val140)+(val51*val183)+(val62*val186)+(val73*val187)+(val84*val220)+(val95*val221)+(val106*val254)+(val117*val255)+(val2*val130)+(val13*val131)+(val24*val132)+(val33*val133)+(val34*val134)+(val35*val135)+(val36*val136)+(val37*val137)+(val38*val138)+(val39*val139)+(val41*val141)+(val42*val142)+(val43*val143)+(val44*val144)+(val45*val145)+(val46*val146)+(val47*val151)+(val48*val162)+(val49*val173)+(val50*val182)+(val52*val184)+(val53*val185));
    buf0[1] = (buf0[1]+(val0*val147)+(val1*val148)+(val40*val149)+(val51*val150)+(val62*val152)+(val73*val153)+(val84*val154)+(val95*val155)+(val106*val156)+(val117*val157)+(val2*val158)+(val13*val159)+(val24*val160)+(val33*val161)+(val34*val163)+(val35*val164)+(val36*val165)+(val37*val166)+(val38*val167)+(val39*val168)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val174)+(val46*val175)+(val47*val176)+(val48*val177)+(val49*val178)+(val50*val179)+(val52*val180)+(val53*val181));
    buf0[10] = (buf0[10]+(val89*val188)+(val90*val189)+(val91*val190)+(val92*val191)+(val93*val192)+(val94*val193)+(val96*val194)+(val97*val195)+(val98*val196)+(val99*val197)+(val100*val198)+(val101*val199)+(val102*val200)+(val103*val201)+(val104*val202)+(val105*val203)+(val107*val204)+(val108*val205)+(val109*val206)+(val110*val207)+(val111*val208)+(val112*val209)+(val113*val210)+(val114*val211)+(val115*val212)+(val116*val213)+(val118*val214)+(val119*val215)+(val120*val216)+(val121*val217)+(val122*val218)+(val123*val219));
    buf0[11] = (buf0[11]+(val89*val222)+(val90*val223)+(val91*val224)+(val92*val225)+(val93*val226)+(val94*val227)+(val96*val228)+(val97*val229)+(val98*val230)+(val99*val231)+(val100*val232)+(val101*val233)+(val102*val234)+(val103*val235)+(val104*val236)+(val105*val237)+(val107*val238)+(val108*val239)+(val109*val240)+(val110*val241)+(val111*val242)+(val112*val243)+(val113*val244)+(val114*val245)+(val115*val246)+(val116*val247)+(val118*val248)+(val119*val249)+(val120*val250)+(val121*val251)+(val122*val252)+(val123*val253));
    buf0[12] = (buf0[12]+(val124*val128)+(val125*val129)+(val126*val140)+(val127*val183)+(val3*val186)+(val4*val187)+(val5*val220)+(val6*val221)+(val7*val254)+(val8*val255)+(val9*val130)+(val10*val131)+(val11*val132)+(val12*val133)+(val14*val134)+(val15*val135)+(val16*val136)+(val17*val137)+(val18*val138)+(val19*val139)+(val20*val141)+(val21*val142)+(val22*val143)+(val23*val144)+(val25*val145)+(val26*val146)+(val27*val151)+(val28*val162)+(val29*val173)+(val30*val182)+(val31*val184)+(val32*val185));
    buf0[13] = (buf0[13]+(val124*val147)+(val125*val148)+(val126*val149)+(val127*val150)+(val3*val152)+(val4*val153)+(val5*val154)+(val6*val155)+(val7*val156)+(val8*val157)+(val9*val158)+(val10*val159)+(val11*val160)+(val12*val161)+(val14*val163)+(val15*val164)+(val16*val165)+(val17*val166)+(val18*val167)+(val19*val168)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val174)+(val26*val175)+(val27*val176)+(val28*val177)+(val29*val178)+(val30*val179)+(val31*val180)+(val32*val181));
    buf0[14] = (buf0[14]+(val124*val188)+(val125*val189)+(val126*val190)+(val127*val191)+(val3*val192)+(val4*val193)+(val5*val194)+(val6*val195)+(val7*val196)+(val8*val197)+(val9*val198)+(val10*val199)+(val11*val200)+(val12*val201)+(val14*val202)+(val15*val203)+(val16*val204)+(val17*val205)+(val18*val206)+(val19*val207)+(val20*val208)+(val21*val209)+(val22*val210)+(val23*val211)+(val25*val212)+(val26*val213)+(val27*val214)+(val28*val215)+(val29*val216)+(val30*val217)+(val31*val218)+(val32*val219));
    buf0[15] = (buf0[15]+(val124*val222)+(val125*val223)+(val126*val224)+(val127*val225)+(val3*val226)+(val4*val227)+(val5*val228)+(val6*val229)+(val7*val230)+(val8*val231)+(val9*val232)+(val10*val233)+(val11*val234)+(val12*val235)+(val14*val236)+(val15*val237)+(val16*val238)+(val17*val239)+(val18*val240)+(val19*val241)+(val20*val242)+(val21*val243)+(val22*val244)+(val23*val245)+(val25*val246)+(val26*val247)+(val27*val248)+(val28*val249)+(val29*val250)+(val30*val251)+(val31*val252)+(val32*val253));
    buf0[2] = (buf0[2]+(val0*val188)+(val1*val189)+(val40*val190)+(val51*val191)+(val62*val192)+(val73*val193)+(val84*val194)+(val95*val195)+(val106*val196)+(val117*val197)+(val2*val198)+(val13*val199)+(val24*val200)+(val33*val201)+(val34*val202)+(val35*val203)+(val36*val204)+(val37*val205)+(val38*val206)+(val39*val207)+(val41*val208)+(val42*val209)+(val43*val210)+(val44*val211)+(val45*val212)+(val46*val213)+(val47*val214)+(val48*val215)+(val49*val216)+(val50*val217)+(val52*val218)+(val53*val219));
    buf0[3] = (buf0[3]+(val0*val222)+(val1*val223)+(val40*val224)+(val51*val225)+(val62*val226)+(val73*val227)+(val84*val228)+(val95*val229)+(val106*val230)+(val117*val231)+(val2*val232)+(val13*val233)+(val24*val234)+(val33*val235)+(val34*val236)+(val35*val237)+(val36*val238)+(val37*val239)+(val38*val240)+(val39*val241)+(val41*val242)+(val42*val243)+(val43*val244)+(val44*val245)+(val45*val246)+(val46*val247)+(val47*val248)+(val48*val249)+(val49*val250)+(val50*val251)+(val52*val252)+(val53*val253));
    buf0[4] = (buf0[4]+(val54*val128)+(val55*val129)+(val56*val140)+(val57*val183)+(val58*val186)+(val59*val187)+(val60*val220)+(val61*val221)+(val63*val254)+(val64*val255)+(val65*val130)+(val66*val131)+(val67*val132)+(val68*val133)+(val69*val134)+(val70*val135)+(val71*val136)+(val72*val137)+(val74*val138)+(val75*val139)+(val76*val141)+(val77*val142)+(val78*val143)+(val79*val144)+(val80*val145)+(val81*val146)+(val82*val151)+(val83*val162)+(val85*val173)+(val86*val182)+(val87*val184)+(val88*val185));
    buf0[5] = (buf0[5]+(val54*val147)+(val55*val148)+(val56*val149)+(val57*val150)+(val58*val152)+(val59*val153)+(val60*val154)+(val61*val155)+(val63*val156)+(val64*val157)+(val65*val158)+(val66*val159)+(val67*val160)+(val68*val161)+(val69*val163)+(val70*val164)+(val71*val165)+(val72*val166)+(val74*val167)+(val75*val168)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val174)+(val81*val175)+(val82*val176)+(val83*val177)+(val85*val178)+(val86*val179)+(val87*val180)+(val88*val181));
    buf0[6] = (buf0[6]+(val54*val188)+(val55*val189)+(val56*val190)+(val57*val191)+(val58*val192)+(val59*val193)+(val60*val194)+(val61*val195)+(val63*val196)+(val64*val197)+(val65*val198)+(val66*val199)+(val67*val200)+(val68*val201)+(val69*val202)+(val70*val203)+(val71*val204)+(val72*val205)+(val74*val206)+(val75*val207)+(val76*val208)+(val77*val209)+(val78*val210)+(val79*val211)+(val80*val212)+(val81*val213)+(val82*val214)+(val83*val215)+(val85*val216)+(val86*val217)+(val87*val218)+(val88*val219));
    buf0[7] = (buf0[7]+(val54*val222)+(val55*val223)+(val56*val224)+(val57*val225)+(val58*val226)+(val59*val227)+(val60*val228)+(val61*val229)+(val63*val230)+(val64*val231)+(val65*val232)+(val66*val233)+(val67*val234)+(val68*val235)+(val69*val236)+(val70*val237)+(val71*val238)+(val72*val239)+(val74*val240)+(val75*val241)+(val76*val242)+(val77*val243)+(val78*val244)+(val79*val245)+(val80*val246)+(val81*val247)+(val82*val248)+(val83*val249)+(val85*val250)+(val86*val251)+(val87*val252)+(val88*val253));
    buf0[8] = (buf0[8]+(val89*val128)+(val90*val129)+(val91*val140)+(val92*val183)+(val93*val186)+(val94*val187)+(val96*val220)+(val97*val221)+(val98*val254)+(val99*val255)+(val100*val130)+(val101*val131)+(val102*val132)+(val103*val133)+(val104*val134)+(val105*val135)+(val107*val136)+(val108*val137)+(val109*val138)+(val110*val139)+(val111*val141)+(val112*val142)+(val113*val143)+(val114*val144)+(val115*val145)+(val116*val146)+(val118*val151)+(val119*val162)+(val120*val173)+(val121*val182)+(val122*val184)+(val123*val185));
    buf0[9] = (buf0[9]+(val89*val147)+(val90*val148)+(val91*val149)+(val92*val150)+(val93*val152)+(val94*val153)+(val96*val154)+(val97*val155)+(val98*val156)+(val99*val157)+(val100*val158)+(val101*val159)+(val102*val160)+(val103*val161)+(val104*val163)+(val105*val164)+(val107*val165)+(val108*val166)+(val109*val167)+(val110*val168)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val174)+(val116*val175)+(val118*val176)+(val119*val177)+(val120*val178)+(val121*val179)+(val122*val180)+(val123*val181));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<13u))+bitcast<i32>((cast2<<10u)));
  var val256 = data1_65536[alu36];
  var alu37 = (alu36+1);
  var val257 = data1_65536[alu37];
  var alu38 = (alu36+2);
  var val258 = data1_65536[alu38];
  var alu39 = (alu36+256);
  var val259 = data1_65536[alu39];
  var alu40 = (alu36+257);
  var val260 = data1_65536[alu40];
  var alu41 = (alu36+258);
  var val261 = data1_65536[alu41];
  var alu42 = (alu36+259);
  var val262 = data1_65536[alu42];
  var alu43 = (alu36+3);
  var val263 = data1_65536[alu43];
  var alu44 = (alu36+512);
  var val264 = data1_65536[alu44];
  var alu45 = (alu36+513);
  var val265 = data1_65536[alu45];
  var alu46 = (alu36+514);
  var val266 = data1_65536[alu46];
  var alu47 = (alu36+515);
  var val267 = data1_65536[alu47];
  var alu48 = (alu36+768);
  var val268 = data1_65536[alu48];
  var alu49 = (alu36+769);
  var val269 = data1_65536[alu49];
  var alu50 = (alu36+770);
  var val270 = data1_65536[alu50];
  var alu51 = (alu36+771);
  var val271 = data1_65536[alu51];
  var val272 = data2_1536[(alu35+512)];
  var val273 = data2_1536[(alu35+513)];
  var val274 = data2_1536[(alu35+514)];
  var val275 = data2_1536[(alu35+515)];
  var val276 = data5_256[alu35];
  var val277 = data5_256[(alu35+1)];
  var val278 = data5_256[(alu35+2)];
  var val279 = data5_256[(alu35+3)];
  data0_65536[alu36] = (val256+(val272*(buf0[0]+val276)));
  data0_65536[alu37] = (val257+(val273*(buf0[1]+val277)));
  data0_65536[alu38] = (val258+(val274*(buf0[2]+val278)));
  data0_65536[alu39] = (val259+(val272*(buf0[4]+val276)));
  data0_65536[alu40] = (val260+(val273*(buf0[5]+val277)));
  data0_65536[alu41] = (val261+(val274*(buf0[6]+val278)));
  data0_65536[alu42] = (val262+(val275*(buf0[7]+val279)));
  data0_65536[alu43] = (val263+(val275*(buf0[3]+val279)));
  data0_65536[alu44] = (val264+(val272*(buf0[8]+val276)));
  data0_65536[alu45] = (val265+(val273*(buf0[9]+val277)));
  data0_65536[alu46] = (val266+(val274*(buf0[10]+val278)));
  data0_65536[alu47] = (val267+(val275*(buf0[11]+val279)));
  data0_65536[alu48] = (val268+(val272*(buf0[12]+val276)));
  data0_65536[alu49] = (val269+(val273*(buf0[13]+val277)));
  data0_65536[alu50] = (val270+(val274*(buf0[14]+val278)));
  data0_65536[alu51] = (val271+(val275*(buf0[15]+val279)));
}`;

const r_16_8_4_4_32_dispatch_25 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_26 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var val21 = data3_256[(alu0+1)];
  var val22 = data3_256[(alu0+2)];
  var val23 = data3_256[(alu0+3)];
  var val24 = data4_1536[(alu0+1024)];
  var val25 = data4_1536[(alu0+1025)];
  var val26 = data4_1536[(alu0+1026)];
  var val27 = data4_1536[(alu0+1027)];
  var val28 = data4_1536[(alu0+768)];
  var val29 = data4_1536[(alu0+769)];
  var val30 = data4_1536[(alu0+770)];
  var val31 = data4_1536[(alu0+771)];
  var alu18 = (1/val16);
  var alu19 = (1/val17);
  var alu20 = (1/val18);
  var alu21 = (1/val19);
  var alu22 = (1.0f+val24);
  var alu23 = (1.0f+val25);
  var alu24 = (1.0f+val26);
  var alu25 = (1.0f+val27);
  data0_65536[alu1] = ((val0*alu18*val20*alu22)+val28);
  data0_65536[alu2] = ((val1*alu18*val21*alu23)+val29);
  data0_65536[alu3] = ((val2*alu18*val22*alu24)+val30);
  data0_65536[alu4] = ((val3*alu19*val20*alu22)+val28);
  data0_65536[alu5] = ((val4*alu19*val21*alu23)+val29);
  data0_65536[alu6] = ((val5*alu19*val22*alu24)+val30);
  data0_65536[alu7] = ((val6*alu19*val23*alu25)+val31);
  data0_65536[alu8] = ((val7*alu18*val23*alu25)+val31);
  data0_65536[alu9] = ((val8*alu20*val20*alu22)+val28);
  data0_65536[alu10] = ((val9*alu20*val21*alu23)+val29);
  data0_65536[alu11] = ((val10*alu20*val22*alu24)+val30);
  data0_65536[alu12] = ((val11*alu20*val23*alu25)+val31);
  data0_65536[alu13] = ((val12*alu21*val20*alu22)+val28);
  data0_65536[alu14] = ((val13*alu21*val21*alu23)+val29);
  data0_65536[alu15] = ((val14*alu21*val22*alu24)+val30);
  data0_65536[alu16] = ((val15*alu21*val23*alu25)+val31);
}`;

const r_8_22_8_16_4_4_64_4_dispatch_27 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_360448:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_360448:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1408:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 22 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast2 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast2);
    var val0 = data1_65536[alu16];
    var val1 = data1_65536[(alu16+1)];
    var val2 = data1_65536[(alu16+2)];
    var val3 = data1_65536[(alu16+256)];
    var val4 = data1_65536[(alu16+257)];
    var val5 = data1_65536[(alu16+258)];
    var val6 = data1_65536[(alu16+259)];
    var val7 = data1_65536[(alu16+3)];
    var val8 = data1_65536[(alu16+512)];
    var val9 = data1_65536[(alu16+513)];
    var val10 = data1_65536[(alu16+514)];
    var val11 = data1_65536[(alu16+515)];
    var val12 = data1_65536[(alu16+768)];
    var val13 = data1_65536[(alu16+769)];
    var val14 = data1_65536[(alu16+770)];
    var val15 = data1_65536[(alu16+771)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast1<<10u))+cast2);
    var val16 = data2_360448[alu17];
    var val17 = data2_360448[(alu17+1)];
    var val18 = data2_360448[(alu17+2)];
    var val19 = data2_360448[(alu17+256)];
    var val20 = data2_360448[(alu17+257)];
    var val21 = data2_360448[(alu17+258)];
    var val22 = data2_360448[(alu17+259)];
    var val23 = data2_360448[(alu17+3)];
    var val24 = data2_360448[(alu17+512)];
    var val25 = data2_360448[(alu17+513)];
    var val26 = data2_360448[(alu17+514)];
    var val27 = data2_360448[(alu17+515)];
    var val28 = data2_360448[(alu17+768)];
    var val29 = data2_360448[(alu17+769)];
    var val30 = data2_360448[(alu17+770)];
    var val31 = data2_360448[(alu17+771)];
    buf0[0] = (buf0[0]+(val0*val16)+(val1*val17)+(val2*val18)+(val7*val23));
    buf0[1] = (buf0[1]+(val0*val19)+(val1*val20)+(val2*val21)+(val7*val22));
    buf0[10] = (buf0[10]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    buf0[11] = (buf0[11]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    buf0[12] = (buf0[12]+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val23));
    buf0[13] = (buf0[13]+(val12*val19)+(val13*val20)+(val14*val21)+(val15*val22));
    buf0[14] = (buf0[14]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    buf0[15] = (buf0[15]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    buf0[2] = (buf0[2]+(val0*val24)+(val1*val25)+(val2*val26)+(val7*val27));
    buf0[3] = (buf0[3]+(val0*val28)+(val1*val29)+(val2*val30)+(val7*val31));
    buf0[4] = (buf0[4]+(val3*val16)+(val4*val17)+(val5*val18)+(val6*val23));
    buf0[5] = (buf0[5]+(val3*val19)+(val4*val20)+(val5*val21)+(val6*val22));
    buf0[6] = (buf0[6]+(val3*val24)+(val4*val25)+(val5*val26)+(val6*val27));
    buf0[7] = (buf0[7]+(val3*val28)+(val4*val29)+(val5*val30)+(val6*val31));
    buf0[8] = (buf0[8]+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val23));
    buf0[9] = (buf0[9]+(val8*val19)+(val9*val20)+(val10*val21)+(val11*val22));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)));
  var val32 = data3_1408[alu35];
  var val33 = data3_1408[(alu35+1)];
  var val34 = data3_1408[(alu35+2)];
  var val35 = data3_1408[(alu35+3)];
  var alu36 = (alu35+(gidx1*45056)+(lidx0*5632));
  data0_360448[alu36] = (buf0[0]+val32);
  data0_360448[(alu36+1)] = (buf0[1]+val33);
  data0_360448[(alu36+1408)] = (buf0[4]+val32);
  data0_360448[(alu36+1409)] = (buf0[5]+val33);
  data0_360448[(alu36+1410)] = (buf0[6]+val34);
  data0_360448[(alu36+1411)] = (buf0[7]+val35);
  data0_360448[(alu36+2)] = (buf0[2]+val34);
  data0_360448[(alu36+2816)] = (buf0[8]+val32);
  data0_360448[(alu36+2817)] = (buf0[9]+val33);
  data0_360448[(alu36+2818)] = (buf0[10]+val34);
  data0_360448[(alu36+2819)] = (buf0[11]+val35);
  data0_360448[(alu36+3)] = (buf0[3]+val35);
  data0_360448[(alu36+4224)] = (buf0[12]+val32);
  data0_360448[(alu36+4225)] = (buf0[13]+val33);
  data0_360448[(alu36+4226)] = (buf0[14]+val34);
  data0_360448[(alu36+4227)] = (buf0[15]+val35);
}`;

const r_8_4_8_16_4_4_176_4_dispatch_28 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_360448:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_180224:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 176; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = ((gidx1*45056)+(lidx0*5632)+cast0);
    var val0 = data3_360448[(alu16+1)];
    var val1 = data3_360448[(alu16+1408)];
    var val2 = data3_360448[(alu16+1409)];
    var val3 = data3_360448[(alu16+1410)];
    var val4 = data3_360448[(alu16+1411)];
    var val5 = data3_360448[(alu16+2)];
    var val6 = data3_360448[(alu16+2112)];
    var val7 = data3_360448[(alu16+2113)];
    var val8 = data3_360448[(alu16+2114)];
    var val9 = data3_360448[(alu16+2115)];
    var val10 = data3_360448[(alu16+2816)];
    var val11 = data3_360448[(alu16+2817)];
    var val12 = data3_360448[(alu16+2818)];
    var val13 = data3_360448[(alu16+2819)];
    var val14 = data3_360448[(alu16+3)];
    var val15 = data3_360448[(alu16+3520)];
    var val16 = data3_360448[(alu16+3521)];
    var val17 = data3_360448[(alu16+3522)];
    var val18 = data3_360448[(alu16+3523)];
    var val19 = data3_360448[(alu16+4224)];
    var val20 = data3_360448[(alu16+4225)];
    var val21 = data3_360448[(alu16+4226)];
    var val22 = data3_360448[(alu16+4227)];
    var val23 = data3_360448[(alu16+4928)];
    var val24 = data3_360448[(alu16+4929)];
    var val25 = data3_360448[(alu16+4930)];
    var val26 = data3_360448[(alu16+4931)];
    var val27 = data3_360448[(alu16+704)];
    var val28 = data3_360448[(alu16+705)];
    var val29 = data3_360448[(alu16+706)];
    var val30 = data3_360448[(alu16+707)];
    var val31 = data3_360448[alu16];
    var alu17 = ((gidx0*45056)+(lidx1*2816)+cast0);
    var val32 = data4_180224[(alu17+1)];
    var val33 = data4_180224[(alu17+1408)];
    var val34 = data4_180224[(alu17+1409)];
    var val35 = data4_180224[(alu17+1410)];
    var val36 = data4_180224[(alu17+1411)];
    var val37 = data4_180224[(alu17+2)];
    var val38 = data4_180224[(alu17+2112)];
    var val39 = data4_180224[(alu17+2113)];
    var val40 = data4_180224[(alu17+2114)];
    var val41 = data4_180224[(alu17+2115)];
    var val42 = data4_180224[(alu17+3)];
    var val43 = data4_180224[(alu17+704)];
    var val44 = data4_180224[(alu17+705)];
    var val45 = data4_180224[(alu17+706)];
    var val46 = data4_180224[(alu17+707)];
    var val47 = data4_180224[alu17];
    var alu18 = (val0*(1/(1.0f+exp2((val0*-1.4426950216293335f))))*val28);
    var alu19 = (val31*(1/(1.0f+exp2((val31*-1.4426950216293335f))))*val27);
    var alu20 = (val5*(1/(1.0f+exp2((val5*-1.4426950216293335f))))*val29);
    var alu21 = (val14*(1/(1.0f+exp2((val14*-1.4426950216293335f))))*val30);
    var alu22 = (val10*(1/(1.0f+exp2((val10*-1.4426950216293335f))))*val15);
    var alu23 = (val11*(1/(1.0f+exp2((val11*-1.4426950216293335f))))*val16);
    var alu24 = (val12*(1/(1.0f+exp2((val12*-1.4426950216293335f))))*val17);
    var alu25 = (val13*(1/(1.0f+exp2((val13*-1.4426950216293335f))))*val18);
    var alu26 = (val19*(1/(1.0f+exp2((val19*-1.4426950216293335f))))*val23);
    var alu27 = (val20*(1/(1.0f+exp2((val20*-1.4426950216293335f))))*val24);
    var alu28 = (val21*(1/(1.0f+exp2((val21*-1.4426950216293335f))))*val25);
    var alu29 = (val22*(1/(1.0f+exp2((val22*-1.4426950216293335f))))*val26);
    var alu30 = (val1*(1/(1.0f+exp2((val1*-1.4426950216293335f))))*val6);
    var alu31 = (val2*(1/(1.0f+exp2((val2*-1.4426950216293335f))))*val7);
    var alu32 = (val3*(1/(1.0f+exp2((val3*-1.4426950216293335f))))*val8);
    var alu33 = (val4*(1/(1.0f+exp2((val4*-1.4426950216293335f))))*val9);
    buf0[0] = (buf0[0]+(alu19*val47)+(alu18*val32)+(alu20*val37)+(alu21*val42));
    buf0[1] = (buf0[1]+(alu19*val43)+(alu18*val44)+(alu20*val45)+(alu21*val46));
    buf0[10] = (buf0[10]+(alu22*val33)+(alu23*val34)+(alu24*val35)+(alu25*val36));
    buf0[11] = (buf0[11]+(alu22*val38)+(alu23*val39)+(alu24*val40)+(alu25*val41));
    buf0[12] = (buf0[12]+(alu26*val47)+(alu27*val32)+(alu28*val37)+(alu29*val42));
    buf0[13] = (buf0[13]+(alu26*val43)+(alu27*val44)+(alu28*val45)+(alu29*val46));
    buf0[14] = (buf0[14]+(alu26*val33)+(alu27*val34)+(alu28*val35)+(alu29*val36));
    buf0[15] = (buf0[15]+(alu26*val38)+(alu27*val39)+(alu28*val40)+(alu29*val41));
    buf0[2] = (buf0[2]+(alu19*val33)+(alu18*val34)+(alu20*val35)+(alu21*val36));
    buf0[3] = (buf0[3]+(alu19*val38)+(alu18*val39)+(alu20*val40)+(alu21*val41));
    buf0[4] = (buf0[4]+(alu30*val47)+(alu31*val32)+(alu32*val37)+(alu33*val42));
    buf0[5] = (buf0[5]+(alu30*val43)+(alu31*val44)+(alu32*val45)+(alu33*val46));
    buf0[6] = (buf0[6]+(alu30*val33)+(alu31*val34)+(alu32*val35)+(alu33*val36));
    buf0[7] = (buf0[7]+(alu30*val38)+(alu31*val39)+(alu32*val40)+(alu33*val41));
    buf0[8] = (buf0[8]+(alu22*val47)+(alu23*val32)+(alu24*val37)+(alu25*val42));
    buf0[9] = (buf0[9]+(alu22*val43)+(alu23*val44)+(alu24*val45)+(alu25*val46));
  }
  var alu51 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu52 = (alu51+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u)));
  var val48 = data1_65536[alu52];
  var alu53 = (alu52+1);
  var val49 = data1_65536[alu53];
  var alu54 = (alu52+2);
  var val50 = data1_65536[alu54];
  var alu55 = (alu52+256);
  var val51 = data1_65536[alu55];
  var alu56 = (alu52+257);
  var val52 = data1_65536[alu56];
  var alu57 = (alu52+258);
  var val53 = data1_65536[alu57];
  var alu58 = (alu52+259);
  var val54 = data1_65536[alu58];
  var alu59 = (alu52+3);
  var val55 = data1_65536[alu59];
  var alu60 = (alu52+512);
  var val56 = data1_65536[alu60];
  var alu61 = (alu52+513);
  var val57 = data1_65536[alu61];
  var alu62 = (alu52+514);
  var val58 = data1_65536[alu62];
  var alu63 = (alu52+515);
  var val59 = data1_65536[alu63];
  var alu64 = (alu52+768);
  var val60 = data1_65536[alu64];
  var alu65 = (alu52+769);
  var val61 = data1_65536[alu65];
  var alu66 = (alu52+770);
  var val62 = data1_65536[alu66];
  var alu67 = (alu52+771);
  var val63 = data1_65536[alu67];
  var val64 = data2_1536[(alu51+1280)];
  var val65 = data2_1536[(alu51+1281)];
  var val66 = data2_1536[(alu51+1282)];
  var val67 = data2_1536[(alu51+1283)];
  var val68 = data5_256[alu51];
  var val69 = data5_256[(alu51+1)];
  var val70 = data5_256[(alu51+2)];
  var val71 = data5_256[(alu51+3)];
  data0_65536[alu52] = (val48+(val64*(buf0[0]+val68)));
  data0_65536[alu53] = (val49+(val65*(buf0[1]+val69)));
  data0_65536[alu54] = (val50+(val66*(buf0[2]+val70)));
  data0_65536[alu55] = (val51+(val64*(buf0[4]+val68)));
  data0_65536[alu56] = (val52+(val65*(buf0[5]+val69)));
  data0_65536[alu57] = (val53+(val66*(buf0[6]+val70)));
  data0_65536[alu58] = (val54+(val67*(buf0[7]+val71)));
  data0_65536[alu59] = (val55+(val67*(buf0[3]+val71)));
  data0_65536[alu60] = (val56+(val64*(buf0[8]+val68)));
  data0_65536[alu61] = (val57+(val65*(buf0[9]+val69)));
  data0_65536[alu62] = (val58+(val66*(buf0[10]+val70)));
  data0_65536[alu63] = (val59+(val67*(buf0[11]+val71)));
  data0_65536[alu64] = (val60+(val64*(buf0[12]+val68)));
  data0_65536[alu65] = (val61+(val65*(buf0[13]+val69)));
  data0_65536[alu66] = (val62+(val66*(buf0[14]+val70)));
  data0_65536[alu67] = (val63+(val67*(buf0[15]+val71)));
}`;

const r_16_8_4_4_32_dispatch_29 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_30 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var alu18 = (alu0+1);
  var val21 = data3_256[alu18];
  var alu19 = (alu0+2);
  var val22 = data3_256[alu19];
  var alu20 = (alu0+3);
  var val23 = data3_256[alu20];
  var val24 = data4_1536[alu0];
  var val25 = data4_1536[alu18];
  var val26 = data4_1536[alu19];
  var val27 = data4_1536[(alu0+256)];
  var val28 = data4_1536[(alu0+257)];
  var val29 = data4_1536[(alu0+258)];
  var val30 = data4_1536[(alu0+259)];
  var val31 = data4_1536[alu20];
  var alu21 = (1/val16);
  var alu22 = (1/val17);
  var alu23 = (1/val18);
  var alu24 = (1/val19);
  var alu25 = (1.0f+val27);
  var alu26 = (1.0f+val28);
  var alu27 = (1.0f+val29);
  var alu28 = (1.0f+val30);
  data0_65536[alu1] = ((val0*alu21*val20*alu25)+val24);
  data0_65536[alu2] = ((val1*alu21*val21*alu26)+val25);
  data0_65536[alu3] = ((val2*alu21*val22*alu27)+val26);
  data0_65536[alu4] = ((val3*alu22*val20*alu25)+val24);
  data0_65536[alu5] = ((val4*alu22*val21*alu26)+val25);
  data0_65536[alu6] = ((val5*alu22*val22*alu27)+val26);
  data0_65536[alu7] = ((val6*alu22*val23*alu28)+val31);
  data0_65536[alu8] = ((val7*alu21*val23*alu28)+val31);
  data0_65536[alu9] = ((val8*alu23*val20*alu25)+val24);
  data0_65536[alu10] = ((val9*alu23*val21*alu26)+val25);
  data0_65536[alu11] = ((val10*alu23*val22*alu27)+val26);
  data0_65536[alu12] = ((val11*alu23*val23*alu28)+val31);
  data0_65536[alu13] = ((val12*alu24*val20*alu25)+val24);
  data0_65536[alu14] = ((val13*alu24*val21*alu26)+val25);
  data0_65536[alu15] = ((val14*alu24*val22*alu27)+val26);
  data0_65536[alu16] = ((val15*alu24*val23*alu28)+val31);
}`;

const r_8_16_8_16_3_4_64_4_dispatch_31 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_196608:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu12 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast0);
    var val0 = data1_65536[alu12];
    var val1 = data1_65536[(alu12+1)];
    var val2 = data1_65536[(alu12+2)];
    var val3 = data1_65536[(alu12+256)];
    var val4 = data1_65536[(alu12+257)];
    var val5 = data1_65536[(alu12+258)];
    var val6 = data1_65536[(alu12+259)];
    var val7 = data1_65536[(alu12+3)];
    var val8 = data1_65536[(alu12+512)];
    var val9 = data1_65536[(alu12+513)];
    var val10 = data1_65536[(alu12+514)];
    var val11 = data1_65536[(alu12+515)];
    var val12 = data1_65536[(alu12+768)];
    var val13 = data1_65536[(alu12+769)];
    var val14 = data1_65536[(alu12+770)];
    var val15 = data1_65536[(alu12+771)];
    var alu13 = ((gidx0*12288)+(lidx1*768)+cast0);
    var val16 = data2_196608[(alu13+1)];
    var val17 = data2_196608[(alu13+2)];
    var val18 = data2_196608[(alu13+256)];
    var val19 = data2_196608[(alu13+257)];
    var val20 = data2_196608[(alu13+258)];
    var val21 = data2_196608[(alu13+259)];
    var val22 = data2_196608[(alu13+3)];
    var val23 = data2_196608[(alu13+512)];
    var val24 = data2_196608[(alu13+513)];
    var val25 = data2_196608[(alu13+514)];
    var val26 = data2_196608[(alu13+515)];
    var val27 = data2_196608[alu13];
    buf0[0] = (buf0[0]+(val0*val27)+(val1*val16)+(val2*val17)+(val7*val22));
    buf0[1] = (buf0[1]+(val0*val18)+(val1*val19)+(val2*val20)+(val7*val21));
    buf0[10] = (buf0[10]+(val12*val18)+(val13*val19)+(val14*val20)+(val15*val21));
    buf0[11] = (buf0[11]+(val12*val23)+(val13*val24)+(val14*val25)+(val15*val26));
    buf0[2] = (buf0[2]+(val0*val23)+(val1*val24)+(val2*val25)+(val7*val26));
    buf0[3] = (buf0[3]+(val3*val27)+(val4*val16)+(val5*val17)+(val6*val22));
    buf0[4] = (buf0[4]+(val3*val18)+(val4*val19)+(val5*val20)+(val6*val21));
    buf0[5] = (buf0[5]+(val3*val23)+(val4*val24)+(val5*val25)+(val6*val26));
    buf0[6] = (buf0[6]+(val8*val27)+(val9*val16)+(val10*val17)+(val11*val22));
    buf0[7] = (buf0[7]+(val8*val18)+(val9*val19)+(val10*val20)+(val11*val21));
    buf0[8] = (buf0[8]+(val8*val23)+(val9*val24)+(val10*val25)+(val11*val26));
    buf0[9] = (buf0[9]+(val12*val27)+(val13*val16)+(val14*val17)+(val15*val22));
  }
  var alu27 = ((gidx0*48)+(lidx1*3));
  var val28 = data3_768[(alu27+1)];
  var val29 = data3_768[(alu27+2)];
  var val30 = data3_768[alu27];
  var alu28 = (alu27+(gidx1*24576)+(lidx0*3072));
  data0_196608[(alu28+1)] = (buf0[1]+val28);
  data0_196608[(alu28+1536)] = (buf0[6]+val30);
  data0_196608[(alu28+1537)] = (buf0[7]+val28);
  data0_196608[(alu28+1538)] = (buf0[8]+val29);
  data0_196608[(alu28+2)] = (buf0[2]+val29);
  data0_196608[(alu28+2304)] = (buf0[9]+val30);
  data0_196608[(alu28+2305)] = (buf0[10]+val28);
  data0_196608[(alu28+2306)] = (buf0[11]+val29);
  data0_196608[(alu28+768)] = (buf0[3]+val30);
  data0_196608[(alu28+769)] = (buf0[4]+val28);
  data0_196608[(alu28+770)] = (buf0[5]+val29);
  data0_196608[alu28] = (buf0[0]+val30);
}`;

const r_8_16_8_4_4_4_dispatch_32 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[alu4];
    var val1 = data1_196608[(alu4+3072)];
    var val2 = data1_196608[(alu4+6144)];
    var val3 = data1_196608[(alu4+9216)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const r_8_16_8_4_4_4_dispatch_33 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[(alu4+256)];
    var val1 = data1_196608[(alu4+3328)];
    var val2 = data1_196608[(alu4+6400)];
    var val3 = data1_196608[(alu4+9472)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const E_2_32_8_16_2_4_dispatch_34 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+1)];
  var val1 = data1_196608[(alu0+32)];
  var val2 = data1_196608[(alu0+33)];
  var val3 = data1_196608[(alu0+64)];
  var val4 = data1_196608[(alu0+65)];
  var val5 = data1_196608[(alu0+96)];
  var val6 = data1_196608[(alu0+97)];
  var val7 = data1_196608[alu0];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val13);
  var alu10 = (val7*alu5*val12);
  var alu11 = (val1*alu6*val12);
  var alu12 = (val2*alu6*val13);
  var alu13 = (val3*alu7*val12);
  var alu14 = (val4*alu7*val13);
  var alu15 = (val5*alu8*val12);
  var alu16 = (val6*alu8*val13);
  data0_65536[alu4] = ((alu10*val14)-(alu9*val16));
  data0_65536[(alu4+1)] = ((alu9*val15)+(alu10*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const E_2_32_8_16_2_4_dispatch_35 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+256)];
  var val1 = data1_196608[(alu0+257)];
  var val2 = data1_196608[(alu0+288)];
  var val3 = data1_196608[(alu0+289)];
  var val4 = data1_196608[(alu0+320)];
  var val5 = data1_196608[(alu0+321)];
  var val6 = data1_196608[(alu0+352)];
  var val7 = data1_196608[(alu0+353)];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val12);
  var alu10 = (val1*alu5*val13);
  var alu11 = (val2*alu6*val12);
  var alu12 = (val3*alu6*val13);
  var alu13 = (val4*alu7*val12);
  var alu14 = (val5*alu7*val13);
  var alu15 = (val6*alu8*val12);
  var alu16 = (val7*alu8*val13);
  data0_65536[alu4] = ((alu9*val14)-(alu10*val16));
  data0_65536[(alu4+1)] = ((alu10*val15)+(alu9*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const r_8_8_4_8_16_4_4_32_dispatch_36 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_524288:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
  var cast2 = bitcast<i32>((cast1<<13u));
  var cast3 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<10u))+bitcast<i32>((cast3<<7u))+cast2);
  var val0 = data1_65536[alu0];
  var val1 = data1_65536[(alu0+1)];
  var val2 = data1_65536[(alu0+10)];
  var val3 = data1_65536[(alu0+100)];
  var val4 = data1_65536[(alu0+101)];
  var val5 = data1_65536[(alu0+102)];
  var val6 = data1_65536[(alu0+103)];
  var val7 = data1_65536[(alu0+104)];
  var val8 = data1_65536[(alu0+105)];
  var val9 = data1_65536[(alu0+106)];
  var val10 = data1_65536[(alu0+107)];
  var val11 = data1_65536[(alu0+108)];
  var val12 = data1_65536[(alu0+109)];
  var val13 = data1_65536[(alu0+11)];
  var val14 = data1_65536[(alu0+110)];
  var val15 = data1_65536[(alu0+111)];
  var val16 = data1_65536[(alu0+112)];
  var val17 = data1_65536[(alu0+113)];
  var val18 = data1_65536[(alu0+114)];
  var val19 = data1_65536[(alu0+115)];
  var val20 = data1_65536[(alu0+116)];
  var val21 = data1_65536[(alu0+117)];
  var val22 = data1_65536[(alu0+118)];
  var val23 = data1_65536[(alu0+119)];
  var val24 = data1_65536[(alu0+12)];
  var val25 = data1_65536[(alu0+120)];
  var val26 = data1_65536[(alu0+121)];
  var val27 = data1_65536[(alu0+122)];
  var val28 = data1_65536[(alu0+123)];
  var val29 = data1_65536[(alu0+124)];
  var val30 = data1_65536[(alu0+125)];
  var val31 = data1_65536[(alu0+126)];
  var val32 = data1_65536[(alu0+127)];
  var val33 = data1_65536[(alu0+13)];
  var val34 = data1_65536[(alu0+14)];
  var val35 = data1_65536[(alu0+15)];
  var val36 = data1_65536[(alu0+16)];
  var val37 = data1_65536[(alu0+17)];
  var val38 = data1_65536[(alu0+18)];
  var val39 = data1_65536[(alu0+19)];
  var val40 = data1_65536[(alu0+2)];
  var val41 = data1_65536[(alu0+20)];
  var val42 = data1_65536[(alu0+21)];
  var val43 = data1_65536[(alu0+22)];
  var val44 = data1_65536[(alu0+23)];
  var val45 = data1_65536[(alu0+24)];
  var val46 = data1_65536[(alu0+25)];
  var val47 = data1_65536[(alu0+26)];
  var val48 = data1_65536[(alu0+27)];
  var val49 = data1_65536[(alu0+28)];
  var val50 = data1_65536[(alu0+29)];
  var val51 = data1_65536[(alu0+3)];
  var val52 = data1_65536[(alu0+30)];
  var val53 = data1_65536[(alu0+31)];
  var val54 = data1_65536[(alu0+32)];
  var val55 = data1_65536[(alu0+33)];
  var val56 = data1_65536[(alu0+34)];
  var val57 = data1_65536[(alu0+35)];
  var val58 = data1_65536[(alu0+36)];
  var val59 = data1_65536[(alu0+37)];
  var val60 = data1_65536[(alu0+38)];
  var val61 = data1_65536[(alu0+39)];
  var val62 = data1_65536[(alu0+4)];
  var val63 = data1_65536[(alu0+40)];
  var val64 = data1_65536[(alu0+41)];
  var val65 = data1_65536[(alu0+42)];
  var val66 = data1_65536[(alu0+43)];
  var val67 = data1_65536[(alu0+44)];
  var val68 = data1_65536[(alu0+45)];
  var val69 = data1_65536[(alu0+46)];
  var val70 = data1_65536[(alu0+47)];
  var val71 = data1_65536[(alu0+48)];
  var val72 = data1_65536[(alu0+49)];
  var val73 = data1_65536[(alu0+5)];
  var val74 = data1_65536[(alu0+50)];
  var val75 = data1_65536[(alu0+51)];
  var val76 = data1_65536[(alu0+52)];
  var val77 = data1_65536[(alu0+53)];
  var val78 = data1_65536[(alu0+54)];
  var val79 = data1_65536[(alu0+55)];
  var val80 = data1_65536[(alu0+56)];
  var val81 = data1_65536[(alu0+57)];
  var val82 = data1_65536[(alu0+58)];
  var val83 = data1_65536[(alu0+59)];
  var val84 = data1_65536[(alu0+6)];
  var val85 = data1_65536[(alu0+60)];
  var val86 = data1_65536[(alu0+61)];
  var val87 = data1_65536[(alu0+62)];
  var val88 = data1_65536[(alu0+63)];
  var val89 = data1_65536[(alu0+64)];
  var val90 = data1_65536[(alu0+65)];
  var val91 = data1_65536[(alu0+66)];
  var val92 = data1_65536[(alu0+67)];
  var val93 = data1_65536[(alu0+68)];
  var val94 = data1_65536[(alu0+69)];
  var val95 = data1_65536[(alu0+7)];
  var val96 = data1_65536[(alu0+70)];
  var val97 = data1_65536[(alu0+71)];
  var val98 = data1_65536[(alu0+72)];
  var val99 = data1_65536[(alu0+73)];
  var val100 = data1_65536[(alu0+74)];
  var val101 = data1_65536[(alu0+75)];
  var val102 = data1_65536[(alu0+76)];
  var val103 = data1_65536[(alu0+77)];
  var val104 = data1_65536[(alu0+78)];
  var val105 = data1_65536[(alu0+79)];
  var val106 = data1_65536[(alu0+8)];
  var val107 = data1_65536[(alu0+80)];
  var val108 = data1_65536[(alu0+81)];
  var val109 = data1_65536[(alu0+82)];
  var val110 = data1_65536[(alu0+83)];
  var val111 = data1_65536[(alu0+84)];
  var val112 = data1_65536[(alu0+85)];
  var val113 = data1_65536[(alu0+86)];
  var val114 = data1_65536[(alu0+87)];
  var val115 = data1_65536[(alu0+88)];
  var val116 = data1_65536[(alu0+89)];
  var val117 = data1_65536[(alu0+9)];
  var val118 = data1_65536[(alu0+90)];
  var val119 = data1_65536[(alu0+91)];
  var val120 = data1_65536[(alu0+92)];
  var val121 = data1_65536[(alu0+93)];
  var val122 = data1_65536[(alu0+94)];
  var val123 = data1_65536[(alu0+95)];
  var val124 = data1_65536[(alu0+96)];
  var val125 = data1_65536[(alu0+97)];
  var val126 = data1_65536[(alu0+98)];
  var val127 = data1_65536[(alu0+99)];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast4 = bitcast<u32>(gidx0);
  var cast5 = bitcast<u32>(lidx1);
  var alu1 = (bitcast<i32>((cast4<<11u))+bitcast<i32>((cast5<<7u))+cast2);
  var val128 = data2_65536[alu1];
  var val129 = data2_65536[(alu1+1)];
  var val130 = data2_65536[(alu1+10)];
  var val131 = data2_65536[(alu1+100)];
  var val132 = data2_65536[(alu1+101)];
  var val133 = data2_65536[(alu1+102)];
  var val134 = data2_65536[(alu1+103)];
  var val135 = data2_65536[(alu1+104)];
  var val136 = data2_65536[(alu1+105)];
  var val137 = data2_65536[(alu1+106)];
  var val138 = data2_65536[(alu1+107)];
  var val139 = data2_65536[(alu1+108)];
  var val140 = data2_65536[(alu1+109)];
  var val141 = data2_65536[(alu1+11)];
  var val142 = data2_65536[(alu1+110)];
  var val143 = data2_65536[(alu1+111)];
  var val144 = data2_65536[(alu1+112)];
  var val145 = data2_65536[(alu1+113)];
  var val146 = data2_65536[(alu1+114)];
  var val147 = data2_65536[(alu1+115)];
  var val148 = data2_65536[(alu1+116)];
  var val149 = data2_65536[(alu1+117)];
  var val150 = data2_65536[(alu1+118)];
  var val151 = data2_65536[(alu1+119)];
  var val152 = data2_65536[(alu1+12)];
  var val153 = data2_65536[(alu1+120)];
  var val154 = data2_65536[(alu1+121)];
  var val155 = data2_65536[(alu1+122)];
  var val156 = data2_65536[(alu1+123)];
  var val157 = data2_65536[(alu1+124)];
  var val158 = data2_65536[(alu1+125)];
  var val159 = data2_65536[(alu1+126)];
  var val160 = data2_65536[(alu1+127)];
  var val161 = data2_65536[(alu1+13)];
  var val162 = data2_65536[(alu1+14)];
  var val163 = data2_65536[(alu1+15)];
  var val164 = data2_65536[(alu1+16)];
  var val165 = data2_65536[(alu1+17)];
  var val166 = data2_65536[(alu1+18)];
  var val167 = data2_65536[(alu1+19)];
  var val168 = data2_65536[(alu1+2)];
  var val169 = data2_65536[(alu1+20)];
  var val170 = data2_65536[(alu1+21)];
  var val171 = data2_65536[(alu1+22)];
  var val172 = data2_65536[(alu1+23)];
  var val173 = data2_65536[(alu1+24)];
  var val174 = data2_65536[(alu1+25)];
  var val175 = data2_65536[(alu1+26)];
  var val176 = data2_65536[(alu1+27)];
  var val177 = data2_65536[(alu1+28)];
  var val178 = data2_65536[(alu1+29)];
  var val179 = data2_65536[(alu1+3)];
  var val180 = data2_65536[(alu1+30)];
  var val181 = data2_65536[(alu1+31)];
  var val182 = data2_65536[(alu1+32)];
  var val183 = data2_65536[(alu1+33)];
  var val184 = data2_65536[(alu1+34)];
  var val185 = data2_65536[(alu1+35)];
  var val186 = data2_65536[(alu1+36)];
  var val187 = data2_65536[(alu1+37)];
  var val188 = data2_65536[(alu1+38)];
  var val189 = data2_65536[(alu1+39)];
  var val190 = data2_65536[(alu1+4)];
  var val191 = data2_65536[(alu1+40)];
  var val192 = data2_65536[(alu1+41)];
  var val193 = data2_65536[(alu1+42)];
  var val194 = data2_65536[(alu1+43)];
  var val195 = data2_65536[(alu1+44)];
  var val196 = data2_65536[(alu1+45)];
  var val197 = data2_65536[(alu1+46)];
  var val198 = data2_65536[(alu1+47)];
  var val199 = data2_65536[(alu1+48)];
  var val200 = data2_65536[(alu1+49)];
  var val201 = data2_65536[(alu1+5)];
  var val202 = data2_65536[(alu1+50)];
  var val203 = data2_65536[(alu1+51)];
  var val204 = data2_65536[(alu1+52)];
  var val205 = data2_65536[(alu1+53)];
  var val206 = data2_65536[(alu1+54)];
  var val207 = data2_65536[(alu1+55)];
  var val208 = data2_65536[(alu1+56)];
  var val209 = data2_65536[(alu1+57)];
  var val210 = data2_65536[(alu1+58)];
  var val211 = data2_65536[(alu1+59)];
  var val212 = data2_65536[(alu1+6)];
  var val213 = data2_65536[(alu1+60)];
  var val214 = data2_65536[(alu1+61)];
  var val215 = data2_65536[(alu1+62)];
  var val216 = data2_65536[(alu1+63)];
  var val217 = data2_65536[(alu1+64)];
  var val218 = data2_65536[(alu1+65)];
  var val219 = data2_65536[(alu1+66)];
  var val220 = data2_65536[(alu1+67)];
  var val221 = data2_65536[(alu1+68)];
  var val222 = data2_65536[(alu1+69)];
  var val223 = data2_65536[(alu1+7)];
  var val224 = data2_65536[(alu1+70)];
  var val225 = data2_65536[(alu1+71)];
  var val226 = data2_65536[(alu1+72)];
  var val227 = data2_65536[(alu1+73)];
  var val228 = data2_65536[(alu1+74)];
  var val229 = data2_65536[(alu1+75)];
  var val230 = data2_65536[(alu1+76)];
  var val231 = data2_65536[(alu1+77)];
  var val232 = data2_65536[(alu1+78)];
  var val233 = data2_65536[(alu1+79)];
  var val234 = data2_65536[(alu1+8)];
  var val235 = data2_65536[(alu1+80)];
  var val236 = data2_65536[(alu1+81)];
  var val237 = data2_65536[(alu1+82)];
  var val238 = data2_65536[(alu1+83)];
  var val239 = data2_65536[(alu1+84)];
  var val240 = data2_65536[(alu1+85)];
  var val241 = data2_65536[(alu1+86)];
  var val242 = data2_65536[(alu1+87)];
  var val243 = data2_65536[(alu1+88)];
  var val244 = data2_65536[(alu1+89)];
  var val245 = data2_65536[(alu1+9)];
  var val246 = data2_65536[(alu1+90)];
  var val247 = data2_65536[(alu1+91)];
  var val248 = data2_65536[(alu1+92)];
  var val249 = data2_65536[(alu1+93)];
  var val250 = data2_65536[(alu1+94)];
  var val251 = data2_65536[(alu1+95)];
  var val252 = data2_65536[(alu1+96)];
  var val253 = data2_65536[(alu1+97)];
  var val254 = data2_65536[(alu1+98)];
  var val255 = data2_65536[(alu1+99)];
  var alu2 = (bitcast<i32>((cast4<<6u))+bitcast<i32>((cast5<<2u))+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast1<<16u)));
  data0_524288[alu2] = (((val0*val128)+(val1*val129)+(val40*val168)+(val51*val179)+(val62*val190)+(val73*val201)+(val84*val212)+(val95*val223)+(val106*val234)+(val117*val245)+(val2*val130)+(val13*val141)+(val24*val152)+(val33*val161)+(val34*val162)+(val35*val163)+(val36*val164)+(val37*val165)+(val38*val166)+(val39*val167)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val173)+(val46*val174)+(val47*val175)+(val48*val176)+(val49*val177)+(val50*val178)+(val52*val180)+(val53*val181))*0.1767766922712326f);
  data0_524288[(alu2+1)] = (((val0*val182)+(val1*val183)+(val40*val184)+(val51*val185)+(val62*val186)+(val73*val187)+(val84*val188)+(val95*val189)+(val106*val191)+(val117*val192)+(val2*val193)+(val13*val194)+(val24*val195)+(val33*val196)+(val34*val197)+(val35*val198)+(val36*val199)+(val37*val200)+(val38*val202)+(val39*val203)+(val41*val204)+(val42*val205)+(val43*val206)+(val44*val207)+(val45*val208)+(val46*val209)+(val47*val210)+(val48*val211)+(val49*val213)+(val50*val214)+(val52*val215)+(val53*val216))*0.1767766922712326f);
  data0_524288[(alu2+2)] = (((val0*val217)+(val1*val218)+(val40*val219)+(val51*val220)+(val62*val221)+(val73*val222)+(val84*val224)+(val95*val225)+(val106*val226)+(val117*val227)+(val2*val228)+(val13*val229)+(val24*val230)+(val33*val231)+(val34*val232)+(val35*val233)+(val36*val235)+(val37*val236)+(val38*val237)+(val39*val238)+(val41*val239)+(val42*val240)+(val43*val241)+(val44*val242)+(val45*val243)+(val46*val244)+(val47*val246)+(val48*val247)+(val49*val248)+(val50*val249)+(val52*val250)+(val53*val251))*0.1767766922712326f);
  data0_524288[(alu2+256)] = (((val54*val128)+(val55*val129)+(val56*val168)+(val57*val179)+(val58*val190)+(val59*val201)+(val60*val212)+(val61*val223)+(val63*val234)+(val64*val245)+(val65*val130)+(val66*val141)+(val67*val152)+(val68*val161)+(val69*val162)+(val70*val163)+(val71*val164)+(val72*val165)+(val74*val166)+(val75*val167)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val173)+(val81*val174)+(val82*val175)+(val83*val176)+(val85*val177)+(val86*val178)+(val87*val180)+(val88*val181))*0.1767766922712326f);
  data0_524288[(alu2+257)] = (((val54*val182)+(val55*val183)+(val56*val184)+(val57*val185)+(val58*val186)+(val59*val187)+(val60*val188)+(val61*val189)+(val63*val191)+(val64*val192)+(val65*val193)+(val66*val194)+(val67*val195)+(val68*val196)+(val69*val197)+(val70*val198)+(val71*val199)+(val72*val200)+(val74*val202)+(val75*val203)+(val76*val204)+(val77*val205)+(val78*val206)+(val79*val207)+(val80*val208)+(val81*val209)+(val82*val210)+(val83*val211)+(val85*val213)+(val86*val214)+(val87*val215)+(val88*val216))*0.1767766922712326f);
  data0_524288[(alu2+258)] = (((val54*val217)+(val55*val218)+(val56*val219)+(val57*val220)+(val58*val221)+(val59*val222)+(val60*val224)+(val61*val225)+(val63*val226)+(val64*val227)+(val65*val228)+(val66*val229)+(val67*val230)+(val68*val231)+(val69*val232)+(val70*val233)+(val71*val235)+(val72*val236)+(val74*val237)+(val75*val238)+(val76*val239)+(val77*val240)+(val78*val241)+(val79*val242)+(val80*val243)+(val81*val244)+(val82*val246)+(val83*val247)+(val85*val248)+(val86*val249)+(val87*val250)+(val88*val251))*0.1767766922712326f);
  data0_524288[(alu2+259)] = (((val54*val252)+(val55*val253)+(val56*val254)+(val57*val255)+(val58*val131)+(val59*val132)+(val60*val133)+(val61*val134)+(val63*val135)+(val64*val136)+(val65*val137)+(val66*val138)+(val67*val139)+(val68*val140)+(val69*val142)+(val70*val143)+(val71*val144)+(val72*val145)+(val74*val146)+(val75*val147)+(val76*val148)+(val77*val149)+(val78*val150)+(val79*val151)+(val80*val153)+(val81*val154)+(val82*val155)+(val83*val156)+(val85*val157)+(val86*val158)+(val87*val159)+(val88*val160))*0.1767766922712326f);
  data0_524288[(alu2+3)] = (((val0*val252)+(val1*val253)+(val40*val254)+(val51*val255)+(val62*val131)+(val73*val132)+(val84*val133)+(val95*val134)+(val106*val135)+(val117*val136)+(val2*val137)+(val13*val138)+(val24*val139)+(val33*val140)+(val34*val142)+(val35*val143)+(val36*val144)+(val37*val145)+(val38*val146)+(val39*val147)+(val41*val148)+(val42*val149)+(val43*val150)+(val44*val151)+(val45*val153)+(val46*val154)+(val47*val155)+(val48*val156)+(val49*val157)+(val50*val158)+(val52*val159)+(val53*val160))*0.1767766922712326f);
  data0_524288[(alu2+512)] = (((val89*val128)+(val90*val129)+(val91*val168)+(val92*val179)+(val93*val190)+(val94*val201)+(val96*val212)+(val97*val223)+(val98*val234)+(val99*val245)+(val100*val130)+(val101*val141)+(val102*val152)+(val103*val161)+(val104*val162)+(val105*val163)+(val107*val164)+(val108*val165)+(val109*val166)+(val110*val167)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val173)+(val116*val174)+(val118*val175)+(val119*val176)+(val120*val177)+(val121*val178)+(val122*val180)+(val123*val181))*0.1767766922712326f);
  data0_524288[(alu2+513)] = (((val89*val182)+(val90*val183)+(val91*val184)+(val92*val185)+(val93*val186)+(val94*val187)+(val96*val188)+(val97*val189)+(val98*val191)+(val99*val192)+(val100*val193)+(val101*val194)+(val102*val195)+(val103*val196)+(val104*val197)+(val105*val198)+(val107*val199)+(val108*val200)+(val109*val202)+(val110*val203)+(val111*val204)+(val112*val205)+(val113*val206)+(val114*val207)+(val115*val208)+(val116*val209)+(val118*val210)+(val119*val211)+(val120*val213)+(val121*val214)+(val122*val215)+(val123*val216))*0.1767766922712326f);
  data0_524288[(alu2+514)] = (((val89*val217)+(val90*val218)+(val91*val219)+(val92*val220)+(val93*val221)+(val94*val222)+(val96*val224)+(val97*val225)+(val98*val226)+(val99*val227)+(val100*val228)+(val101*val229)+(val102*val230)+(val103*val231)+(val104*val232)+(val105*val233)+(val107*val235)+(val108*val236)+(val109*val237)+(val110*val238)+(val111*val239)+(val112*val240)+(val113*val241)+(val114*val242)+(val115*val243)+(val116*val244)+(val118*val246)+(val119*val247)+(val120*val248)+(val121*val249)+(val122*val250)+(val123*val251))*0.1767766922712326f);
  data0_524288[(alu2+515)] = (((val89*val252)+(val90*val253)+(val91*val254)+(val92*val255)+(val93*val131)+(val94*val132)+(val96*val133)+(val97*val134)+(val98*val135)+(val99*val136)+(val100*val137)+(val101*val138)+(val102*val139)+(val103*val140)+(val104*val142)+(val105*val143)+(val107*val144)+(val108*val145)+(val109*val146)+(val110*val147)+(val111*val148)+(val112*val149)+(val113*val150)+(val114*val151)+(val115*val153)+(val116*val154)+(val118*val155)+(val119*val156)+(val120*val157)+(val121*val158)+(val122*val159)+(val123*val160))*0.1767766922712326f);
  data0_524288[(alu2+768)] = (((val124*val128)+(val125*val129)+(val126*val168)+(val127*val179)+(val3*val190)+(val4*val201)+(val5*val212)+(val6*val223)+(val7*val234)+(val8*val245)+(val9*val130)+(val10*val141)+(val11*val152)+(val12*val161)+(val14*val162)+(val15*val163)+(val16*val164)+(val17*val165)+(val18*val166)+(val19*val167)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val173)+(val26*val174)+(val27*val175)+(val28*val176)+(val29*val177)+(val30*val178)+(val31*val180)+(val32*val181))*0.1767766922712326f);
  data0_524288[(alu2+769)] = (((val124*val182)+(val125*val183)+(val126*val184)+(val127*val185)+(val3*val186)+(val4*val187)+(val5*val188)+(val6*val189)+(val7*val191)+(val8*val192)+(val9*val193)+(val10*val194)+(val11*val195)+(val12*val196)+(val14*val197)+(val15*val198)+(val16*val199)+(val17*val200)+(val18*val202)+(val19*val203)+(val20*val204)+(val21*val205)+(val22*val206)+(val23*val207)+(val25*val208)+(val26*val209)+(val27*val210)+(val28*val211)+(val29*val213)+(val30*val214)+(val31*val215)+(val32*val216))*0.1767766922712326f);
  data0_524288[(alu2+770)] = (((val124*val217)+(val125*val218)+(val126*val219)+(val127*val220)+(val3*val221)+(val4*val222)+(val5*val224)+(val6*val225)+(val7*val226)+(val8*val227)+(val9*val228)+(val10*val229)+(val11*val230)+(val12*val231)+(val14*val232)+(val15*val233)+(val16*val235)+(val17*val236)+(val18*val237)+(val19*val238)+(val20*val239)+(val21*val240)+(val22*val241)+(val23*val242)+(val25*val243)+(val26*val244)+(val27*val246)+(val28*val247)+(val29*val248)+(val30*val249)+(val31*val250)+(val32*val251))*0.1767766922712326f);
  data0_524288[(alu2+771)] = (((val124*val252)+(val125*val253)+(val126*val254)+(val127*val255)+(val3*val131)+(val4*val132)+(val5*val133)+(val6*val134)+(val7*val135)+(val8*val136)+(val9*val137)+(val10*val138)+(val11*val139)+(val12*val140)+(val14*val142)+(val15*val143)+(val16*val144)+(val17*val145)+(val18*val146)+(val19*val147)+(val20*val148)+(val21*val149)+(val22*val150)+(val23*val151)+(val25*val153)+(val26*val154)+(val27*val155)+(val28*val156)+(val29*val157)+(val30*val158)+(val31*val159)+(val32*val160))*0.1767766922712326f);
}`;

const r_2048_16_16_dispatch_37 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = (f32(-INFINITY));
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val0 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    var val1 = buf0[0];
    var alu1 = select(val1,val0,(val1<val0));
    buf0[0] = alu1;
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = (f32(-INFINITY));
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf1[0];
    var val3 = buf2[Lidx102];
    var alu7 = select(val2,val3,(val2<val3));
    buf1[0] = alu7;
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_2048_16_16_dispatch_38 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var val0 = data2_2048[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val1 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    buf0[0] = (buf0[0]+exp2(((val1-val0)*1.4426950216293335f)));
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf2[Lidx102];
    buf1[0] = (buf1[0]+val2);
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_8_4_16_8_4_4_64_4_dispatch_39 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2048:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_196608:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u))+bitcast<i32>((cast1<<8u)));
  var val0 = data2_2048[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_2048[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_2048[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_2048[alu3];
  var lidx1 = i32(lindex.y); /* 8 */
  var cast3 = bitcast<i32>((bitcast<u32>(lidx1)<<2u));
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var alu20 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast2<<10u))+bitcast<i32>((bitcast<u32>(Lidx0)<<2u))+bitcast<i32>((cast1<<16u)));
    var val4 = data1_524288[alu20];
    var val5 = data1_524288[(alu20+1)];
    var val6 = data1_524288[(alu20+2)];
    var val7 = data1_524288[(alu20+256)];
    var val8 = data1_524288[(alu20+257)];
    var val9 = data1_524288[(alu20+258)];
    var val10 = data1_524288[(alu20+259)];
    var val11 = data1_524288[(alu20+3)];
    var val12 = data1_524288[(alu20+512)];
    var val13 = data1_524288[(alu20+513)];
    var val14 = data1_524288[(alu20+514)];
    var val15 = data1_524288[(alu20+515)];
    var val16 = data1_524288[(alu20+768)];
    var val17 = data1_524288[(alu20+769)];
    var val18 = data1_524288[(alu20+770)];
    var val19 = data1_524288[(alu20+771)];
    var alu21 = (bitcast<i32>((cast1<<5u))+cast3+(Lidx0*3072));
    var val20 = data4_196608[(alu21+1280)];
    var val21 = data4_196608[(alu21+1281)];
    var val22 = data4_196608[(alu21+1282)];
    var val23 = data4_196608[(alu21+1283)];
    var val24 = data4_196608[(alu21+2048)];
    var val25 = data4_196608[(alu21+2049)];
    var val26 = data4_196608[(alu21+2050)];
    var val27 = data4_196608[(alu21+2051)];
    var val28 = data4_196608[(alu21+2816)];
    var val29 = data4_196608[(alu21+2817)];
    var val30 = data4_196608[(alu21+2818)];
    var val31 = data4_196608[(alu21+2819)];
    var val32 = data4_196608[(alu21+512)];
    var val33 = data4_196608[(alu21+513)];
    var val34 = data4_196608[(alu21+514)];
    var val35 = data4_196608[(alu21+515)];
    var alu22 = exp2(((val4-val0)*1.4426950216293335f));
    var alu23 = exp2(((val5-val0)*1.4426950216293335f));
    var alu24 = exp2(((val6-val0)*1.4426950216293335f));
    var alu25 = exp2(((val7-val1)*1.4426950216293335f));
    var alu26 = exp2(((val8-val1)*1.4426950216293335f));
    var alu27 = exp2(((val9-val1)*1.4426950216293335f));
    var alu28 = exp2(((val10-val1)*1.4426950216293335f));
    var alu29 = exp2(((val11-val0)*1.4426950216293335f));
    var alu30 = exp2(((val12-val2)*1.4426950216293335f));
    var alu31 = exp2(((val13-val2)*1.4426950216293335f));
    var alu32 = exp2(((val14-val2)*1.4426950216293335f));
    var alu33 = exp2(((val15-val2)*1.4426950216293335f));
    var alu34 = exp2(((val16-val3)*1.4426950216293335f));
    var alu35 = exp2(((val17-val3)*1.4426950216293335f));
    var alu36 = exp2(((val18-val3)*1.4426950216293335f));
    var alu37 = exp2(((val19-val3)*1.4426950216293335f));
    buf0[0] = (buf0[0]+(alu22*val32)+(alu23*val20)+(alu24*val24)+(alu29*val28));
    buf0[1] = (buf0[1]+(alu22*val33)+(alu23*val21)+(alu24*val25)+(alu29*val29));
    buf0[10] = (buf0[10]+(alu30*val34)+(alu31*val22)+(alu32*val26)+(alu33*val30));
    buf0[11] = (buf0[11]+(alu30*val35)+(alu31*val23)+(alu32*val27)+(alu33*val31));
    buf0[12] = (buf0[12]+(alu34*val32)+(alu35*val20)+(alu36*val24)+(alu37*val28));
    buf0[13] = (buf0[13]+(alu34*val33)+(alu35*val21)+(alu36*val25)+(alu37*val29));
    buf0[14] = (buf0[14]+(alu34*val34)+(alu35*val22)+(alu36*val26)+(alu37*val30));
    buf0[15] = (buf0[15]+(alu34*val35)+(alu35*val23)+(alu36*val27)+(alu37*val31));
    buf0[2] = (buf0[2]+(alu22*val34)+(alu23*val22)+(alu24*val26)+(alu29*val30));
    buf0[3] = (buf0[3]+(alu22*val35)+(alu23*val23)+(alu24*val27)+(alu29*val31));
    buf0[4] = (buf0[4]+(alu25*val32)+(alu26*val20)+(alu27*val24)+(alu28*val28));
    buf0[5] = (buf0[5]+(alu25*val33)+(alu26*val21)+(alu27*val25)+(alu28*val29));
    buf0[6] = (buf0[6]+(alu25*val34)+(alu26*val22)+(alu27*val26)+(alu28*val30));
    buf0[7] = (buf0[7]+(alu25*val35)+(alu26*val23)+(alu27*val27)+(alu28*val31));
    buf0[8] = (buf0[8]+(alu30*val32)+(alu31*val20)+(alu32*val24)+(alu33*val28));
    buf0[9] = (buf0[9]+(alu30*val33)+(alu31*val21)+(alu32*val25)+(alu33*val29));
  }
  var val36 = data3_2048[alu0];
  var val37 = data3_2048[alu1];
  var val38 = data3_2048[alu2];
  var val39 = data3_2048[alu3];
  var alu55 = (bitcast<i32>((cast0<<11u))+bitcast<i32>((cast2<<7u))+cast3+bitcast<i32>((cast1<<13u)));
  var alu56 = (1/val36);
  var alu57 = (1/val37);
  var alu58 = (1/val38);
  var alu59 = (1/val39);
  data0_65536[alu55] = (buf0[0]*alu56);
  data0_65536[(alu55+1)] = (buf0[1]*alu56);
  data0_65536[(alu55+2)] = (buf0[2]*alu56);
  data0_65536[(alu55+3)] = (buf0[3]*alu56);
  data0_65536[(alu55+32)] = (buf0[4]*alu57);
  data0_65536[(alu55+33)] = (buf0[5]*alu57);
  data0_65536[(alu55+34)] = (buf0[6]*alu57);
  data0_65536[(alu55+35)] = (buf0[7]*alu57);
  data0_65536[(alu55+64)] = (buf0[8]*alu58);
  data0_65536[(alu55+65)] = (buf0[9]*alu58);
  data0_65536[(alu55+66)] = (buf0[10]*alu58);
  data0_65536[(alu55+67)] = (buf0[11]*alu58);
  data0_65536[(alu55+96)] = (buf0[12]*alu59);
  data0_65536[(alu55+97)] = (buf0[13]*alu59);
  data0_65536[(alu55+98)] = (buf0[14]*alu59);
  data0_65536[(alu55+99)] = (buf0[15]*alu59);
}`;

const r_8_4_8_16_4_4_8_32_dispatch_40 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_65536:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0_0 = 0; Lidx0_0 < 8; Lidx0_0++) {
    var cast4 = bitcast<u32>(Lidx0_0);
    var alu16 = (bitcast<i32>((cast1<<10u))+bitcast<i32>((cast2<<7u))+bitcast<i32>((cast4<<13u)));
    var val0 = data3_65536[alu16];
    var val1 = data3_65536[(alu16+1)];
    var val2 = data3_65536[(alu16+10)];
    var val3 = data3_65536[(alu16+100)];
    var val4 = data3_65536[(alu16+101)];
    var val5 = data3_65536[(alu16+102)];
    var val6 = data3_65536[(alu16+103)];
    var val7 = data3_65536[(alu16+104)];
    var val8 = data3_65536[(alu16+105)];
    var val9 = data3_65536[(alu16+106)];
    var val10 = data3_65536[(alu16+107)];
    var val11 = data3_65536[(alu16+108)];
    var val12 = data3_65536[(alu16+109)];
    var val13 = data3_65536[(alu16+11)];
    var val14 = data3_65536[(alu16+110)];
    var val15 = data3_65536[(alu16+111)];
    var val16 = data3_65536[(alu16+112)];
    var val17 = data3_65536[(alu16+113)];
    var val18 = data3_65536[(alu16+114)];
    var val19 = data3_65536[(alu16+115)];
    var val20 = data3_65536[(alu16+116)];
    var val21 = data3_65536[(alu16+117)];
    var val22 = data3_65536[(alu16+118)];
    var val23 = data3_65536[(alu16+119)];
    var val24 = data3_65536[(alu16+12)];
    var val25 = data3_65536[(alu16+120)];
    var val26 = data3_65536[(alu16+121)];
    var val27 = data3_65536[(alu16+122)];
    var val28 = data3_65536[(alu16+123)];
    var val29 = data3_65536[(alu16+124)];
    var val30 = data3_65536[(alu16+125)];
    var val31 = data3_65536[(alu16+126)];
    var val32 = data3_65536[(alu16+127)];
    var val33 = data3_65536[(alu16+13)];
    var val34 = data3_65536[(alu16+14)];
    var val35 = data3_65536[(alu16+15)];
    var val36 = data3_65536[(alu16+16)];
    var val37 = data3_65536[(alu16+17)];
    var val38 = data3_65536[(alu16+18)];
    var val39 = data3_65536[(alu16+19)];
    var val40 = data3_65536[(alu16+2)];
    var val41 = data3_65536[(alu16+20)];
    var val42 = data3_65536[(alu16+21)];
    var val43 = data3_65536[(alu16+22)];
    var val44 = data3_65536[(alu16+23)];
    var val45 = data3_65536[(alu16+24)];
    var val46 = data3_65536[(alu16+25)];
    var val47 = data3_65536[(alu16+26)];
    var val48 = data3_65536[(alu16+27)];
    var val49 = data3_65536[(alu16+28)];
    var val50 = data3_65536[(alu16+29)];
    var val51 = data3_65536[(alu16+3)];
    var val52 = data3_65536[(alu16+30)];
    var val53 = data3_65536[(alu16+31)];
    var val54 = data3_65536[(alu16+32)];
    var val55 = data3_65536[(alu16+33)];
    var val56 = data3_65536[(alu16+34)];
    var val57 = data3_65536[(alu16+35)];
    var val58 = data3_65536[(alu16+36)];
    var val59 = data3_65536[(alu16+37)];
    var val60 = data3_65536[(alu16+38)];
    var val61 = data3_65536[(alu16+39)];
    var val62 = data3_65536[(alu16+4)];
    var val63 = data3_65536[(alu16+40)];
    var val64 = data3_65536[(alu16+41)];
    var val65 = data3_65536[(alu16+42)];
    var val66 = data3_65536[(alu16+43)];
    var val67 = data3_65536[(alu16+44)];
    var val68 = data3_65536[(alu16+45)];
    var val69 = data3_65536[(alu16+46)];
    var val70 = data3_65536[(alu16+47)];
    var val71 = data3_65536[(alu16+48)];
    var val72 = data3_65536[(alu16+49)];
    var val73 = data3_65536[(alu16+5)];
    var val74 = data3_65536[(alu16+50)];
    var val75 = data3_65536[(alu16+51)];
    var val76 = data3_65536[(alu16+52)];
    var val77 = data3_65536[(alu16+53)];
    var val78 = data3_65536[(alu16+54)];
    var val79 = data3_65536[(alu16+55)];
    var val80 = data3_65536[(alu16+56)];
    var val81 = data3_65536[(alu16+57)];
    var val82 = data3_65536[(alu16+58)];
    var val83 = data3_65536[(alu16+59)];
    var val84 = data3_65536[(alu16+6)];
    var val85 = data3_65536[(alu16+60)];
    var val86 = data3_65536[(alu16+61)];
    var val87 = data3_65536[(alu16+62)];
    var val88 = data3_65536[(alu16+63)];
    var val89 = data3_65536[(alu16+64)];
    var val90 = data3_65536[(alu16+65)];
    var val91 = data3_65536[(alu16+66)];
    var val92 = data3_65536[(alu16+67)];
    var val93 = data3_65536[(alu16+68)];
    var val94 = data3_65536[(alu16+69)];
    var val95 = data3_65536[(alu16+7)];
    var val96 = data3_65536[(alu16+70)];
    var val97 = data3_65536[(alu16+71)];
    var val98 = data3_65536[(alu16+72)];
    var val99 = data3_65536[(alu16+73)];
    var val100 = data3_65536[(alu16+74)];
    var val101 = data3_65536[(alu16+75)];
    var val102 = data3_65536[(alu16+76)];
    var val103 = data3_65536[(alu16+77)];
    var val104 = data3_65536[(alu16+78)];
    var val105 = data3_65536[(alu16+79)];
    var val106 = data3_65536[(alu16+8)];
    var val107 = data3_65536[(alu16+80)];
    var val108 = data3_65536[(alu16+81)];
    var val109 = data3_65536[(alu16+82)];
    var val110 = data3_65536[(alu16+83)];
    var val111 = data3_65536[(alu16+84)];
    var val112 = data3_65536[(alu16+85)];
    var val113 = data3_65536[(alu16+86)];
    var val114 = data3_65536[(alu16+87)];
    var val115 = data3_65536[(alu16+88)];
    var val116 = data3_65536[(alu16+89)];
    var val117 = data3_65536[(alu16+9)];
    var val118 = data3_65536[(alu16+90)];
    var val119 = data3_65536[(alu16+91)];
    var val120 = data3_65536[(alu16+92)];
    var val121 = data3_65536[(alu16+93)];
    var val122 = data3_65536[(alu16+94)];
    var val123 = data3_65536[(alu16+95)];
    var val124 = data3_65536[(alu16+96)];
    var val125 = data3_65536[(alu16+97)];
    var val126 = data3_65536[(alu16+98)];
    var val127 = data3_65536[(alu16+99)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast4<<5u)));
    var val128 = data4_65536[alu17];
    var val129 = data4_65536[(alu17+1)];
    var val130 = data4_65536[(alu17+10)];
    var val131 = data4_65536[(alu17+11)];
    var val132 = data4_65536[(alu17+12)];
    var val133 = data4_65536[(alu17+13)];
    var val134 = data4_65536[(alu17+14)];
    var val135 = data4_65536[(alu17+15)];
    var val136 = data4_65536[(alu17+16)];
    var val137 = data4_65536[(alu17+17)];
    var val138 = data4_65536[(alu17+18)];
    var val139 = data4_65536[(alu17+19)];
    var val140 = data4_65536[(alu17+2)];
    var val141 = data4_65536[(alu17+20)];
    var val142 = data4_65536[(alu17+21)];
    var val143 = data4_65536[(alu17+22)];
    var val144 = data4_65536[(alu17+23)];
    var val145 = data4_65536[(alu17+24)];
    var val146 = data4_65536[(alu17+25)];
    var val147 = data4_65536[(alu17+256)];
    var val148 = data4_65536[(alu17+257)];
    var val149 = data4_65536[(alu17+258)];
    var val150 = data4_65536[(alu17+259)];
    var val151 = data4_65536[(alu17+26)];
    var val152 = data4_65536[(alu17+260)];
    var val153 = data4_65536[(alu17+261)];
    var val154 = data4_65536[(alu17+262)];
    var val155 = data4_65536[(alu17+263)];
    var val156 = data4_65536[(alu17+264)];
    var val157 = data4_65536[(alu17+265)];
    var val158 = data4_65536[(alu17+266)];
    var val159 = data4_65536[(alu17+267)];
    var val160 = data4_65536[(alu17+268)];
    var val161 = data4_65536[(alu17+269)];
    var val162 = data4_65536[(alu17+27)];
    var val163 = data4_65536[(alu17+270)];
    var val164 = data4_65536[(alu17+271)];
    var val165 = data4_65536[(alu17+272)];
    var val166 = data4_65536[(alu17+273)];
    var val167 = data4_65536[(alu17+274)];
    var val168 = data4_65536[(alu17+275)];
    var val169 = data4_65536[(alu17+276)];
    var val170 = data4_65536[(alu17+277)];
    var val171 = data4_65536[(alu17+278)];
    var val172 = data4_65536[(alu17+279)];
    var val173 = data4_65536[(alu17+28)];
    var val174 = data4_65536[(alu17+280)];
    var val175 = data4_65536[(alu17+281)];
    var val176 = data4_65536[(alu17+282)];
    var val177 = data4_65536[(alu17+283)];
    var val178 = data4_65536[(alu17+284)];
    var val179 = data4_65536[(alu17+285)];
    var val180 = data4_65536[(alu17+286)];
    var val181 = data4_65536[(alu17+287)];
    var val182 = data4_65536[(alu17+29)];
    var val183 = data4_65536[(alu17+3)];
    var val184 = data4_65536[(alu17+30)];
    var val185 = data4_65536[(alu17+31)];
    var val186 = data4_65536[(alu17+4)];
    var val187 = data4_65536[(alu17+5)];
    var val188 = data4_65536[(alu17+512)];
    var val189 = data4_65536[(alu17+513)];
    var val190 = data4_65536[(alu17+514)];
    var val191 = data4_65536[(alu17+515)];
    var val192 = data4_65536[(alu17+516)];
    var val193 = data4_65536[(alu17+517)];
    var val194 = data4_65536[(alu17+518)];
    var val195 = data4_65536[(alu17+519)];
    var val196 = data4_65536[(alu17+520)];
    var val197 = data4_65536[(alu17+521)];
    var val198 = data4_65536[(alu17+522)];
    var val199 = data4_65536[(alu17+523)];
    var val200 = data4_65536[(alu17+524)];
    var val201 = data4_65536[(alu17+525)];
    var val202 = data4_65536[(alu17+526)];
    var val203 = data4_65536[(alu17+527)];
    var val204 = data4_65536[(alu17+528)];
    var val205 = data4_65536[(alu17+529)];
    var val206 = data4_65536[(alu17+530)];
    var val207 = data4_65536[(alu17+531)];
    var val208 = data4_65536[(alu17+532)];
    var val209 = data4_65536[(alu17+533)];
    var val210 = data4_65536[(alu17+534)];
    var val211 = data4_65536[(alu17+535)];
    var val212 = data4_65536[(alu17+536)];
    var val213 = data4_65536[(alu17+537)];
    var val214 = data4_65536[(alu17+538)];
    var val215 = data4_65536[(alu17+539)];
    var val216 = data4_65536[(alu17+540)];
    var val217 = data4_65536[(alu17+541)];
    var val218 = data4_65536[(alu17+542)];
    var val219 = data4_65536[(alu17+543)];
    var val220 = data4_65536[(alu17+6)];
    var val221 = data4_65536[(alu17+7)];
    var val222 = data4_65536[(alu17+768)];
    var val223 = data4_65536[(alu17+769)];
    var val224 = data4_65536[(alu17+770)];
    var val225 = data4_65536[(alu17+771)];
    var val226 = data4_65536[(alu17+772)];
    var val227 = data4_65536[(alu17+773)];
    var val228 = data4_65536[(alu17+774)];
    var val229 = data4_65536[(alu17+775)];
    var val230 = data4_65536[(alu17+776)];
    var val231 = data4_65536[(alu17+777)];
    var val232 = data4_65536[(alu17+778)];
    var val233 = data4_65536[(alu17+779)];
    var val234 = data4_65536[(alu17+780)];
    var val235 = data4_65536[(alu17+781)];
    var val236 = data4_65536[(alu17+782)];
    var val237 = data4_65536[(alu17+783)];
    var val238 = data4_65536[(alu17+784)];
    var val239 = data4_65536[(alu17+785)];
    var val240 = data4_65536[(alu17+786)];
    var val241 = data4_65536[(alu17+787)];
    var val242 = data4_65536[(alu17+788)];
    var val243 = data4_65536[(alu17+789)];
    var val244 = data4_65536[(alu17+790)];
    var val245 = data4_65536[(alu17+791)];
    var val246 = data4_65536[(alu17+792)];
    var val247 = data4_65536[(alu17+793)];
    var val248 = data4_65536[(alu17+794)];
    var val249 = data4_65536[(alu17+795)];
    var val250 = data4_65536[(alu17+796)];
    var val251 = data4_65536[(alu17+797)];
    var val252 = data4_65536[(alu17+798)];
    var val253 = data4_65536[(alu17+799)];
    var val254 = data4_65536[(alu17+8)];
    var val255 = data4_65536[(alu17+9)];
    buf0[0] = (buf0[0]+(val0*val128)+(val1*val129)+(val40*val140)+(val51*val183)+(val62*val186)+(val73*val187)+(val84*val220)+(val95*val221)+(val106*val254)+(val117*val255)+(val2*val130)+(val13*val131)+(val24*val132)+(val33*val133)+(val34*val134)+(val35*val135)+(val36*val136)+(val37*val137)+(val38*val138)+(val39*val139)+(val41*val141)+(val42*val142)+(val43*val143)+(val44*val144)+(val45*val145)+(val46*val146)+(val47*val151)+(val48*val162)+(val49*val173)+(val50*val182)+(val52*val184)+(val53*val185));
    buf0[1] = (buf0[1]+(val0*val147)+(val1*val148)+(val40*val149)+(val51*val150)+(val62*val152)+(val73*val153)+(val84*val154)+(val95*val155)+(val106*val156)+(val117*val157)+(val2*val158)+(val13*val159)+(val24*val160)+(val33*val161)+(val34*val163)+(val35*val164)+(val36*val165)+(val37*val166)+(val38*val167)+(val39*val168)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val174)+(val46*val175)+(val47*val176)+(val48*val177)+(val49*val178)+(val50*val179)+(val52*val180)+(val53*val181));
    buf0[10] = (buf0[10]+(val89*val188)+(val90*val189)+(val91*val190)+(val92*val191)+(val93*val192)+(val94*val193)+(val96*val194)+(val97*val195)+(val98*val196)+(val99*val197)+(val100*val198)+(val101*val199)+(val102*val200)+(val103*val201)+(val104*val202)+(val105*val203)+(val107*val204)+(val108*val205)+(val109*val206)+(val110*val207)+(val111*val208)+(val112*val209)+(val113*val210)+(val114*val211)+(val115*val212)+(val116*val213)+(val118*val214)+(val119*val215)+(val120*val216)+(val121*val217)+(val122*val218)+(val123*val219));
    buf0[11] = (buf0[11]+(val89*val222)+(val90*val223)+(val91*val224)+(val92*val225)+(val93*val226)+(val94*val227)+(val96*val228)+(val97*val229)+(val98*val230)+(val99*val231)+(val100*val232)+(val101*val233)+(val102*val234)+(val103*val235)+(val104*val236)+(val105*val237)+(val107*val238)+(val108*val239)+(val109*val240)+(val110*val241)+(val111*val242)+(val112*val243)+(val113*val244)+(val114*val245)+(val115*val246)+(val116*val247)+(val118*val248)+(val119*val249)+(val120*val250)+(val121*val251)+(val122*val252)+(val123*val253));
    buf0[12] = (buf0[12]+(val124*val128)+(val125*val129)+(val126*val140)+(val127*val183)+(val3*val186)+(val4*val187)+(val5*val220)+(val6*val221)+(val7*val254)+(val8*val255)+(val9*val130)+(val10*val131)+(val11*val132)+(val12*val133)+(val14*val134)+(val15*val135)+(val16*val136)+(val17*val137)+(val18*val138)+(val19*val139)+(val20*val141)+(val21*val142)+(val22*val143)+(val23*val144)+(val25*val145)+(val26*val146)+(val27*val151)+(val28*val162)+(val29*val173)+(val30*val182)+(val31*val184)+(val32*val185));
    buf0[13] = (buf0[13]+(val124*val147)+(val125*val148)+(val126*val149)+(val127*val150)+(val3*val152)+(val4*val153)+(val5*val154)+(val6*val155)+(val7*val156)+(val8*val157)+(val9*val158)+(val10*val159)+(val11*val160)+(val12*val161)+(val14*val163)+(val15*val164)+(val16*val165)+(val17*val166)+(val18*val167)+(val19*val168)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val174)+(val26*val175)+(val27*val176)+(val28*val177)+(val29*val178)+(val30*val179)+(val31*val180)+(val32*val181));
    buf0[14] = (buf0[14]+(val124*val188)+(val125*val189)+(val126*val190)+(val127*val191)+(val3*val192)+(val4*val193)+(val5*val194)+(val6*val195)+(val7*val196)+(val8*val197)+(val9*val198)+(val10*val199)+(val11*val200)+(val12*val201)+(val14*val202)+(val15*val203)+(val16*val204)+(val17*val205)+(val18*val206)+(val19*val207)+(val20*val208)+(val21*val209)+(val22*val210)+(val23*val211)+(val25*val212)+(val26*val213)+(val27*val214)+(val28*val215)+(val29*val216)+(val30*val217)+(val31*val218)+(val32*val219));
    buf0[15] = (buf0[15]+(val124*val222)+(val125*val223)+(val126*val224)+(val127*val225)+(val3*val226)+(val4*val227)+(val5*val228)+(val6*val229)+(val7*val230)+(val8*val231)+(val9*val232)+(val10*val233)+(val11*val234)+(val12*val235)+(val14*val236)+(val15*val237)+(val16*val238)+(val17*val239)+(val18*val240)+(val19*val241)+(val20*val242)+(val21*val243)+(val22*val244)+(val23*val245)+(val25*val246)+(val26*val247)+(val27*val248)+(val28*val249)+(val29*val250)+(val30*val251)+(val31*val252)+(val32*val253));
    buf0[2] = (buf0[2]+(val0*val188)+(val1*val189)+(val40*val190)+(val51*val191)+(val62*val192)+(val73*val193)+(val84*val194)+(val95*val195)+(val106*val196)+(val117*val197)+(val2*val198)+(val13*val199)+(val24*val200)+(val33*val201)+(val34*val202)+(val35*val203)+(val36*val204)+(val37*val205)+(val38*val206)+(val39*val207)+(val41*val208)+(val42*val209)+(val43*val210)+(val44*val211)+(val45*val212)+(val46*val213)+(val47*val214)+(val48*val215)+(val49*val216)+(val50*val217)+(val52*val218)+(val53*val219));
    buf0[3] = (buf0[3]+(val0*val222)+(val1*val223)+(val40*val224)+(val51*val225)+(val62*val226)+(val73*val227)+(val84*val228)+(val95*val229)+(val106*val230)+(val117*val231)+(val2*val232)+(val13*val233)+(val24*val234)+(val33*val235)+(val34*val236)+(val35*val237)+(val36*val238)+(val37*val239)+(val38*val240)+(val39*val241)+(val41*val242)+(val42*val243)+(val43*val244)+(val44*val245)+(val45*val246)+(val46*val247)+(val47*val248)+(val48*val249)+(val49*val250)+(val50*val251)+(val52*val252)+(val53*val253));
    buf0[4] = (buf0[4]+(val54*val128)+(val55*val129)+(val56*val140)+(val57*val183)+(val58*val186)+(val59*val187)+(val60*val220)+(val61*val221)+(val63*val254)+(val64*val255)+(val65*val130)+(val66*val131)+(val67*val132)+(val68*val133)+(val69*val134)+(val70*val135)+(val71*val136)+(val72*val137)+(val74*val138)+(val75*val139)+(val76*val141)+(val77*val142)+(val78*val143)+(val79*val144)+(val80*val145)+(val81*val146)+(val82*val151)+(val83*val162)+(val85*val173)+(val86*val182)+(val87*val184)+(val88*val185));
    buf0[5] = (buf0[5]+(val54*val147)+(val55*val148)+(val56*val149)+(val57*val150)+(val58*val152)+(val59*val153)+(val60*val154)+(val61*val155)+(val63*val156)+(val64*val157)+(val65*val158)+(val66*val159)+(val67*val160)+(val68*val161)+(val69*val163)+(val70*val164)+(val71*val165)+(val72*val166)+(val74*val167)+(val75*val168)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val174)+(val81*val175)+(val82*val176)+(val83*val177)+(val85*val178)+(val86*val179)+(val87*val180)+(val88*val181));
    buf0[6] = (buf0[6]+(val54*val188)+(val55*val189)+(val56*val190)+(val57*val191)+(val58*val192)+(val59*val193)+(val60*val194)+(val61*val195)+(val63*val196)+(val64*val197)+(val65*val198)+(val66*val199)+(val67*val200)+(val68*val201)+(val69*val202)+(val70*val203)+(val71*val204)+(val72*val205)+(val74*val206)+(val75*val207)+(val76*val208)+(val77*val209)+(val78*val210)+(val79*val211)+(val80*val212)+(val81*val213)+(val82*val214)+(val83*val215)+(val85*val216)+(val86*val217)+(val87*val218)+(val88*val219));
    buf0[7] = (buf0[7]+(val54*val222)+(val55*val223)+(val56*val224)+(val57*val225)+(val58*val226)+(val59*val227)+(val60*val228)+(val61*val229)+(val63*val230)+(val64*val231)+(val65*val232)+(val66*val233)+(val67*val234)+(val68*val235)+(val69*val236)+(val70*val237)+(val71*val238)+(val72*val239)+(val74*val240)+(val75*val241)+(val76*val242)+(val77*val243)+(val78*val244)+(val79*val245)+(val80*val246)+(val81*val247)+(val82*val248)+(val83*val249)+(val85*val250)+(val86*val251)+(val87*val252)+(val88*val253));
    buf0[8] = (buf0[8]+(val89*val128)+(val90*val129)+(val91*val140)+(val92*val183)+(val93*val186)+(val94*val187)+(val96*val220)+(val97*val221)+(val98*val254)+(val99*val255)+(val100*val130)+(val101*val131)+(val102*val132)+(val103*val133)+(val104*val134)+(val105*val135)+(val107*val136)+(val108*val137)+(val109*val138)+(val110*val139)+(val111*val141)+(val112*val142)+(val113*val143)+(val114*val144)+(val115*val145)+(val116*val146)+(val118*val151)+(val119*val162)+(val120*val173)+(val121*val182)+(val122*val184)+(val123*val185));
    buf0[9] = (buf0[9]+(val89*val147)+(val90*val148)+(val91*val149)+(val92*val150)+(val93*val152)+(val94*val153)+(val96*val154)+(val97*val155)+(val98*val156)+(val99*val157)+(val100*val158)+(val101*val159)+(val102*val160)+(val103*val161)+(val104*val163)+(val105*val164)+(val107*val165)+(val108*val166)+(val109*val167)+(val110*val168)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val174)+(val116*val175)+(val118*val176)+(val119*val177)+(val120*val178)+(val121*val179)+(val122*val180)+(val123*val181));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<13u))+bitcast<i32>((cast2<<10u)));
  var val256 = data1_65536[alu36];
  var alu37 = (alu36+1);
  var val257 = data1_65536[alu37];
  var alu38 = (alu36+2);
  var val258 = data1_65536[alu38];
  var alu39 = (alu36+256);
  var val259 = data1_65536[alu39];
  var alu40 = (alu36+257);
  var val260 = data1_65536[alu40];
  var alu41 = (alu36+258);
  var val261 = data1_65536[alu41];
  var alu42 = (alu36+259);
  var val262 = data1_65536[alu42];
  var alu43 = (alu36+3);
  var val263 = data1_65536[alu43];
  var alu44 = (alu36+512);
  var val264 = data1_65536[alu44];
  var alu45 = (alu36+513);
  var val265 = data1_65536[alu45];
  var alu46 = (alu36+514);
  var val266 = data1_65536[alu46];
  var alu47 = (alu36+515);
  var val267 = data1_65536[alu47];
  var alu48 = (alu36+768);
  var val268 = data1_65536[alu48];
  var alu49 = (alu36+769);
  var val269 = data1_65536[alu49];
  var alu50 = (alu36+770);
  var val270 = data1_65536[alu50];
  var alu51 = (alu36+771);
  var val271 = data1_65536[alu51];
  var val272 = data2_1536[(alu35+512)];
  var val273 = data2_1536[(alu35+513)];
  var val274 = data2_1536[(alu35+514)];
  var val275 = data2_1536[(alu35+515)];
  var val276 = data5_256[alu35];
  var val277 = data5_256[(alu35+1)];
  var val278 = data5_256[(alu35+2)];
  var val279 = data5_256[(alu35+3)];
  data0_65536[alu36] = (val256+(val272*(buf0[0]+val276)));
  data0_65536[alu37] = (val257+(val273*(buf0[1]+val277)));
  data0_65536[alu38] = (val258+(val274*(buf0[2]+val278)));
  data0_65536[alu39] = (val259+(val272*(buf0[4]+val276)));
  data0_65536[alu40] = (val260+(val273*(buf0[5]+val277)));
  data0_65536[alu41] = (val261+(val274*(buf0[6]+val278)));
  data0_65536[alu42] = (val262+(val275*(buf0[7]+val279)));
  data0_65536[alu43] = (val263+(val275*(buf0[3]+val279)));
  data0_65536[alu44] = (val264+(val272*(buf0[8]+val276)));
  data0_65536[alu45] = (val265+(val273*(buf0[9]+val277)));
  data0_65536[alu46] = (val266+(val274*(buf0[10]+val278)));
  data0_65536[alu47] = (val267+(val275*(buf0[11]+val279)));
  data0_65536[alu48] = (val268+(val272*(buf0[12]+val276)));
  data0_65536[alu49] = (val269+(val273*(buf0[13]+val277)));
  data0_65536[alu50] = (val270+(val274*(buf0[14]+val278)));
  data0_65536[alu51] = (val271+(val275*(buf0[15]+val279)));
}`;

const r_16_8_4_4_32_dispatch_41 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_42 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var val21 = data3_256[(alu0+1)];
  var val22 = data3_256[(alu0+2)];
  var val23 = data3_256[(alu0+3)];
  var val24 = data4_1536[(alu0+1024)];
  var val25 = data4_1536[(alu0+1025)];
  var val26 = data4_1536[(alu0+1026)];
  var val27 = data4_1536[(alu0+1027)];
  var val28 = data4_1536[(alu0+768)];
  var val29 = data4_1536[(alu0+769)];
  var val30 = data4_1536[(alu0+770)];
  var val31 = data4_1536[(alu0+771)];
  var alu18 = (1/val16);
  var alu19 = (1/val17);
  var alu20 = (1/val18);
  var alu21 = (1/val19);
  var alu22 = (1.0f+val24);
  var alu23 = (1.0f+val25);
  var alu24 = (1.0f+val26);
  var alu25 = (1.0f+val27);
  data0_65536[alu1] = ((val0*alu18*val20*alu22)+val28);
  data0_65536[alu2] = ((val1*alu18*val21*alu23)+val29);
  data0_65536[alu3] = ((val2*alu18*val22*alu24)+val30);
  data0_65536[alu4] = ((val3*alu19*val20*alu22)+val28);
  data0_65536[alu5] = ((val4*alu19*val21*alu23)+val29);
  data0_65536[alu6] = ((val5*alu19*val22*alu24)+val30);
  data0_65536[alu7] = ((val6*alu19*val23*alu25)+val31);
  data0_65536[alu8] = ((val7*alu18*val23*alu25)+val31);
  data0_65536[alu9] = ((val8*alu20*val20*alu22)+val28);
  data0_65536[alu10] = ((val9*alu20*val21*alu23)+val29);
  data0_65536[alu11] = ((val10*alu20*val22*alu24)+val30);
  data0_65536[alu12] = ((val11*alu20*val23*alu25)+val31);
  data0_65536[alu13] = ((val12*alu21*val20*alu22)+val28);
  data0_65536[alu14] = ((val13*alu21*val21*alu23)+val29);
  data0_65536[alu15] = ((val14*alu21*val22*alu24)+val30);
  data0_65536[alu16] = ((val15*alu21*val23*alu25)+val31);
}`;

const r_8_22_8_16_4_4_64_4_dispatch_43 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_360448:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_360448:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1408:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 22 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast2 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast2);
    var val0 = data1_65536[alu16];
    var val1 = data1_65536[(alu16+1)];
    var val2 = data1_65536[(alu16+2)];
    var val3 = data1_65536[(alu16+256)];
    var val4 = data1_65536[(alu16+257)];
    var val5 = data1_65536[(alu16+258)];
    var val6 = data1_65536[(alu16+259)];
    var val7 = data1_65536[(alu16+3)];
    var val8 = data1_65536[(alu16+512)];
    var val9 = data1_65536[(alu16+513)];
    var val10 = data1_65536[(alu16+514)];
    var val11 = data1_65536[(alu16+515)];
    var val12 = data1_65536[(alu16+768)];
    var val13 = data1_65536[(alu16+769)];
    var val14 = data1_65536[(alu16+770)];
    var val15 = data1_65536[(alu16+771)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast1<<10u))+cast2);
    var val16 = data2_360448[alu17];
    var val17 = data2_360448[(alu17+1)];
    var val18 = data2_360448[(alu17+2)];
    var val19 = data2_360448[(alu17+256)];
    var val20 = data2_360448[(alu17+257)];
    var val21 = data2_360448[(alu17+258)];
    var val22 = data2_360448[(alu17+259)];
    var val23 = data2_360448[(alu17+3)];
    var val24 = data2_360448[(alu17+512)];
    var val25 = data2_360448[(alu17+513)];
    var val26 = data2_360448[(alu17+514)];
    var val27 = data2_360448[(alu17+515)];
    var val28 = data2_360448[(alu17+768)];
    var val29 = data2_360448[(alu17+769)];
    var val30 = data2_360448[(alu17+770)];
    var val31 = data2_360448[(alu17+771)];
    buf0[0] = (buf0[0]+(val0*val16)+(val1*val17)+(val2*val18)+(val7*val23));
    buf0[1] = (buf0[1]+(val0*val19)+(val1*val20)+(val2*val21)+(val7*val22));
    buf0[10] = (buf0[10]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    buf0[11] = (buf0[11]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    buf0[12] = (buf0[12]+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val23));
    buf0[13] = (buf0[13]+(val12*val19)+(val13*val20)+(val14*val21)+(val15*val22));
    buf0[14] = (buf0[14]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    buf0[15] = (buf0[15]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    buf0[2] = (buf0[2]+(val0*val24)+(val1*val25)+(val2*val26)+(val7*val27));
    buf0[3] = (buf0[3]+(val0*val28)+(val1*val29)+(val2*val30)+(val7*val31));
    buf0[4] = (buf0[4]+(val3*val16)+(val4*val17)+(val5*val18)+(val6*val23));
    buf0[5] = (buf0[5]+(val3*val19)+(val4*val20)+(val5*val21)+(val6*val22));
    buf0[6] = (buf0[6]+(val3*val24)+(val4*val25)+(val5*val26)+(val6*val27));
    buf0[7] = (buf0[7]+(val3*val28)+(val4*val29)+(val5*val30)+(val6*val31));
    buf0[8] = (buf0[8]+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val23));
    buf0[9] = (buf0[9]+(val8*val19)+(val9*val20)+(val10*val21)+(val11*val22));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)));
  var val32 = data3_1408[alu35];
  var val33 = data3_1408[(alu35+1)];
  var val34 = data3_1408[(alu35+2)];
  var val35 = data3_1408[(alu35+3)];
  var alu36 = (alu35+(gidx1*45056)+(lidx0*5632));
  data0_360448[alu36] = (buf0[0]+val32);
  data0_360448[(alu36+1)] = (buf0[1]+val33);
  data0_360448[(alu36+1408)] = (buf0[4]+val32);
  data0_360448[(alu36+1409)] = (buf0[5]+val33);
  data0_360448[(alu36+1410)] = (buf0[6]+val34);
  data0_360448[(alu36+1411)] = (buf0[7]+val35);
  data0_360448[(alu36+2)] = (buf0[2]+val34);
  data0_360448[(alu36+2816)] = (buf0[8]+val32);
  data0_360448[(alu36+2817)] = (buf0[9]+val33);
  data0_360448[(alu36+2818)] = (buf0[10]+val34);
  data0_360448[(alu36+2819)] = (buf0[11]+val35);
  data0_360448[(alu36+3)] = (buf0[3]+val35);
  data0_360448[(alu36+4224)] = (buf0[12]+val32);
  data0_360448[(alu36+4225)] = (buf0[13]+val33);
  data0_360448[(alu36+4226)] = (buf0[14]+val34);
  data0_360448[(alu36+4227)] = (buf0[15]+val35);
}`;

const r_8_4_8_16_4_4_176_4_dispatch_44 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_360448:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_180224:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 176; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = ((gidx1*45056)+(lidx0*5632)+cast0);
    var val0 = data3_360448[(alu16+1)];
    var val1 = data3_360448[(alu16+1408)];
    var val2 = data3_360448[(alu16+1409)];
    var val3 = data3_360448[(alu16+1410)];
    var val4 = data3_360448[(alu16+1411)];
    var val5 = data3_360448[(alu16+2)];
    var val6 = data3_360448[(alu16+2112)];
    var val7 = data3_360448[(alu16+2113)];
    var val8 = data3_360448[(alu16+2114)];
    var val9 = data3_360448[(alu16+2115)];
    var val10 = data3_360448[(alu16+2816)];
    var val11 = data3_360448[(alu16+2817)];
    var val12 = data3_360448[(alu16+2818)];
    var val13 = data3_360448[(alu16+2819)];
    var val14 = data3_360448[(alu16+3)];
    var val15 = data3_360448[(alu16+3520)];
    var val16 = data3_360448[(alu16+3521)];
    var val17 = data3_360448[(alu16+3522)];
    var val18 = data3_360448[(alu16+3523)];
    var val19 = data3_360448[(alu16+4224)];
    var val20 = data3_360448[(alu16+4225)];
    var val21 = data3_360448[(alu16+4226)];
    var val22 = data3_360448[(alu16+4227)];
    var val23 = data3_360448[(alu16+4928)];
    var val24 = data3_360448[(alu16+4929)];
    var val25 = data3_360448[(alu16+4930)];
    var val26 = data3_360448[(alu16+4931)];
    var val27 = data3_360448[(alu16+704)];
    var val28 = data3_360448[(alu16+705)];
    var val29 = data3_360448[(alu16+706)];
    var val30 = data3_360448[(alu16+707)];
    var val31 = data3_360448[alu16];
    var alu17 = ((gidx0*45056)+(lidx1*2816)+cast0);
    var val32 = data4_180224[(alu17+1)];
    var val33 = data4_180224[(alu17+1408)];
    var val34 = data4_180224[(alu17+1409)];
    var val35 = data4_180224[(alu17+1410)];
    var val36 = data4_180224[(alu17+1411)];
    var val37 = data4_180224[(alu17+2)];
    var val38 = data4_180224[(alu17+2112)];
    var val39 = data4_180224[(alu17+2113)];
    var val40 = data4_180224[(alu17+2114)];
    var val41 = data4_180224[(alu17+2115)];
    var val42 = data4_180224[(alu17+3)];
    var val43 = data4_180224[(alu17+704)];
    var val44 = data4_180224[(alu17+705)];
    var val45 = data4_180224[(alu17+706)];
    var val46 = data4_180224[(alu17+707)];
    var val47 = data4_180224[alu17];
    var alu18 = (val0*(1/(1.0f+exp2((val0*-1.4426950216293335f))))*val28);
    var alu19 = (val31*(1/(1.0f+exp2((val31*-1.4426950216293335f))))*val27);
    var alu20 = (val5*(1/(1.0f+exp2((val5*-1.4426950216293335f))))*val29);
    var alu21 = (val14*(1/(1.0f+exp2((val14*-1.4426950216293335f))))*val30);
    var alu22 = (val10*(1/(1.0f+exp2((val10*-1.4426950216293335f))))*val15);
    var alu23 = (val11*(1/(1.0f+exp2((val11*-1.4426950216293335f))))*val16);
    var alu24 = (val12*(1/(1.0f+exp2((val12*-1.4426950216293335f))))*val17);
    var alu25 = (val13*(1/(1.0f+exp2((val13*-1.4426950216293335f))))*val18);
    var alu26 = (val19*(1/(1.0f+exp2((val19*-1.4426950216293335f))))*val23);
    var alu27 = (val20*(1/(1.0f+exp2((val20*-1.4426950216293335f))))*val24);
    var alu28 = (val21*(1/(1.0f+exp2((val21*-1.4426950216293335f))))*val25);
    var alu29 = (val22*(1/(1.0f+exp2((val22*-1.4426950216293335f))))*val26);
    var alu30 = (val1*(1/(1.0f+exp2((val1*-1.4426950216293335f))))*val6);
    var alu31 = (val2*(1/(1.0f+exp2((val2*-1.4426950216293335f))))*val7);
    var alu32 = (val3*(1/(1.0f+exp2((val3*-1.4426950216293335f))))*val8);
    var alu33 = (val4*(1/(1.0f+exp2((val4*-1.4426950216293335f))))*val9);
    buf0[0] = (buf0[0]+(alu19*val47)+(alu18*val32)+(alu20*val37)+(alu21*val42));
    buf0[1] = (buf0[1]+(alu19*val43)+(alu18*val44)+(alu20*val45)+(alu21*val46));
    buf0[10] = (buf0[10]+(alu22*val33)+(alu23*val34)+(alu24*val35)+(alu25*val36));
    buf0[11] = (buf0[11]+(alu22*val38)+(alu23*val39)+(alu24*val40)+(alu25*val41));
    buf0[12] = (buf0[12]+(alu26*val47)+(alu27*val32)+(alu28*val37)+(alu29*val42));
    buf0[13] = (buf0[13]+(alu26*val43)+(alu27*val44)+(alu28*val45)+(alu29*val46));
    buf0[14] = (buf0[14]+(alu26*val33)+(alu27*val34)+(alu28*val35)+(alu29*val36));
    buf0[15] = (buf0[15]+(alu26*val38)+(alu27*val39)+(alu28*val40)+(alu29*val41));
    buf0[2] = (buf0[2]+(alu19*val33)+(alu18*val34)+(alu20*val35)+(alu21*val36));
    buf0[3] = (buf0[3]+(alu19*val38)+(alu18*val39)+(alu20*val40)+(alu21*val41));
    buf0[4] = (buf0[4]+(alu30*val47)+(alu31*val32)+(alu32*val37)+(alu33*val42));
    buf0[5] = (buf0[5]+(alu30*val43)+(alu31*val44)+(alu32*val45)+(alu33*val46));
    buf0[6] = (buf0[6]+(alu30*val33)+(alu31*val34)+(alu32*val35)+(alu33*val36));
    buf0[7] = (buf0[7]+(alu30*val38)+(alu31*val39)+(alu32*val40)+(alu33*val41));
    buf0[8] = (buf0[8]+(alu22*val47)+(alu23*val32)+(alu24*val37)+(alu25*val42));
    buf0[9] = (buf0[9]+(alu22*val43)+(alu23*val44)+(alu24*val45)+(alu25*val46));
  }
  var alu51 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu52 = (alu51+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u)));
  var val48 = data1_65536[alu52];
  var alu53 = (alu52+1);
  var val49 = data1_65536[alu53];
  var alu54 = (alu52+2);
  var val50 = data1_65536[alu54];
  var alu55 = (alu52+256);
  var val51 = data1_65536[alu55];
  var alu56 = (alu52+257);
  var val52 = data1_65536[alu56];
  var alu57 = (alu52+258);
  var val53 = data1_65536[alu57];
  var alu58 = (alu52+259);
  var val54 = data1_65536[alu58];
  var alu59 = (alu52+3);
  var val55 = data1_65536[alu59];
  var alu60 = (alu52+512);
  var val56 = data1_65536[alu60];
  var alu61 = (alu52+513);
  var val57 = data1_65536[alu61];
  var alu62 = (alu52+514);
  var val58 = data1_65536[alu62];
  var alu63 = (alu52+515);
  var val59 = data1_65536[alu63];
  var alu64 = (alu52+768);
  var val60 = data1_65536[alu64];
  var alu65 = (alu52+769);
  var val61 = data1_65536[alu65];
  var alu66 = (alu52+770);
  var val62 = data1_65536[alu66];
  var alu67 = (alu52+771);
  var val63 = data1_65536[alu67];
  var val64 = data2_1536[(alu51+1280)];
  var val65 = data2_1536[(alu51+1281)];
  var val66 = data2_1536[(alu51+1282)];
  var val67 = data2_1536[(alu51+1283)];
  var val68 = data5_256[alu51];
  var val69 = data5_256[(alu51+1)];
  var val70 = data5_256[(alu51+2)];
  var val71 = data5_256[(alu51+3)];
  data0_65536[alu52] = (val48+(val64*(buf0[0]+val68)));
  data0_65536[alu53] = (val49+(val65*(buf0[1]+val69)));
  data0_65536[alu54] = (val50+(val66*(buf0[2]+val70)));
  data0_65536[alu55] = (val51+(val64*(buf0[4]+val68)));
  data0_65536[alu56] = (val52+(val65*(buf0[5]+val69)));
  data0_65536[alu57] = (val53+(val66*(buf0[6]+val70)));
  data0_65536[alu58] = (val54+(val67*(buf0[7]+val71)));
  data0_65536[alu59] = (val55+(val67*(buf0[3]+val71)));
  data0_65536[alu60] = (val56+(val64*(buf0[8]+val68)));
  data0_65536[alu61] = (val57+(val65*(buf0[9]+val69)));
  data0_65536[alu62] = (val58+(val66*(buf0[10]+val70)));
  data0_65536[alu63] = (val59+(val67*(buf0[11]+val71)));
  data0_65536[alu64] = (val60+(val64*(buf0[12]+val68)));
  data0_65536[alu65] = (val61+(val65*(buf0[13]+val69)));
  data0_65536[alu66] = (val62+(val66*(buf0[14]+val70)));
  data0_65536[alu67] = (val63+(val67*(buf0[15]+val71)));
}`;

const r_16_8_4_4_32_dispatch_45 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_46 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var alu18 = (alu0+1);
  var val21 = data3_256[alu18];
  var alu19 = (alu0+2);
  var val22 = data3_256[alu19];
  var alu20 = (alu0+3);
  var val23 = data3_256[alu20];
  var val24 = data4_1536[alu0];
  var val25 = data4_1536[alu18];
  var val26 = data4_1536[alu19];
  var val27 = data4_1536[(alu0+256)];
  var val28 = data4_1536[(alu0+257)];
  var val29 = data4_1536[(alu0+258)];
  var val30 = data4_1536[(alu0+259)];
  var val31 = data4_1536[alu20];
  var alu21 = (1/val16);
  var alu22 = (1/val17);
  var alu23 = (1/val18);
  var alu24 = (1/val19);
  var alu25 = (1.0f+val27);
  var alu26 = (1.0f+val28);
  var alu27 = (1.0f+val29);
  var alu28 = (1.0f+val30);
  data0_65536[alu1] = ((val0*alu21*val20*alu25)+val24);
  data0_65536[alu2] = ((val1*alu21*val21*alu26)+val25);
  data0_65536[alu3] = ((val2*alu21*val22*alu27)+val26);
  data0_65536[alu4] = ((val3*alu22*val20*alu25)+val24);
  data0_65536[alu5] = ((val4*alu22*val21*alu26)+val25);
  data0_65536[alu6] = ((val5*alu22*val22*alu27)+val26);
  data0_65536[alu7] = ((val6*alu22*val23*alu28)+val31);
  data0_65536[alu8] = ((val7*alu21*val23*alu28)+val31);
  data0_65536[alu9] = ((val8*alu23*val20*alu25)+val24);
  data0_65536[alu10] = ((val9*alu23*val21*alu26)+val25);
  data0_65536[alu11] = ((val10*alu23*val22*alu27)+val26);
  data0_65536[alu12] = ((val11*alu23*val23*alu28)+val31);
  data0_65536[alu13] = ((val12*alu24*val20*alu25)+val24);
  data0_65536[alu14] = ((val13*alu24*val21*alu26)+val25);
  data0_65536[alu15] = ((val14*alu24*val22*alu27)+val26);
  data0_65536[alu16] = ((val15*alu24*val23*alu28)+val31);
}`;

const r_8_16_8_16_3_4_64_4_dispatch_47 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_196608:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu12 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast0);
    var val0 = data1_65536[alu12];
    var val1 = data1_65536[(alu12+1)];
    var val2 = data1_65536[(alu12+2)];
    var val3 = data1_65536[(alu12+256)];
    var val4 = data1_65536[(alu12+257)];
    var val5 = data1_65536[(alu12+258)];
    var val6 = data1_65536[(alu12+259)];
    var val7 = data1_65536[(alu12+3)];
    var val8 = data1_65536[(alu12+512)];
    var val9 = data1_65536[(alu12+513)];
    var val10 = data1_65536[(alu12+514)];
    var val11 = data1_65536[(alu12+515)];
    var val12 = data1_65536[(alu12+768)];
    var val13 = data1_65536[(alu12+769)];
    var val14 = data1_65536[(alu12+770)];
    var val15 = data1_65536[(alu12+771)];
    var alu13 = ((gidx0*12288)+(lidx1*768)+cast0);
    var val16 = data2_196608[(alu13+1)];
    var val17 = data2_196608[(alu13+2)];
    var val18 = data2_196608[(alu13+256)];
    var val19 = data2_196608[(alu13+257)];
    var val20 = data2_196608[(alu13+258)];
    var val21 = data2_196608[(alu13+259)];
    var val22 = data2_196608[(alu13+3)];
    var val23 = data2_196608[(alu13+512)];
    var val24 = data2_196608[(alu13+513)];
    var val25 = data2_196608[(alu13+514)];
    var val26 = data2_196608[(alu13+515)];
    var val27 = data2_196608[alu13];
    buf0[0] = (buf0[0]+(val0*val27)+(val1*val16)+(val2*val17)+(val7*val22));
    buf0[1] = (buf0[1]+(val0*val18)+(val1*val19)+(val2*val20)+(val7*val21));
    buf0[10] = (buf0[10]+(val12*val18)+(val13*val19)+(val14*val20)+(val15*val21));
    buf0[11] = (buf0[11]+(val12*val23)+(val13*val24)+(val14*val25)+(val15*val26));
    buf0[2] = (buf0[2]+(val0*val23)+(val1*val24)+(val2*val25)+(val7*val26));
    buf0[3] = (buf0[3]+(val3*val27)+(val4*val16)+(val5*val17)+(val6*val22));
    buf0[4] = (buf0[4]+(val3*val18)+(val4*val19)+(val5*val20)+(val6*val21));
    buf0[5] = (buf0[5]+(val3*val23)+(val4*val24)+(val5*val25)+(val6*val26));
    buf0[6] = (buf0[6]+(val8*val27)+(val9*val16)+(val10*val17)+(val11*val22));
    buf0[7] = (buf0[7]+(val8*val18)+(val9*val19)+(val10*val20)+(val11*val21));
    buf0[8] = (buf0[8]+(val8*val23)+(val9*val24)+(val10*val25)+(val11*val26));
    buf0[9] = (buf0[9]+(val12*val27)+(val13*val16)+(val14*val17)+(val15*val22));
  }
  var alu27 = ((gidx0*48)+(lidx1*3));
  var val28 = data3_768[(alu27+1)];
  var val29 = data3_768[(alu27+2)];
  var val30 = data3_768[alu27];
  var alu28 = (alu27+(gidx1*24576)+(lidx0*3072));
  data0_196608[(alu28+1)] = (buf0[1]+val28);
  data0_196608[(alu28+1536)] = (buf0[6]+val30);
  data0_196608[(alu28+1537)] = (buf0[7]+val28);
  data0_196608[(alu28+1538)] = (buf0[8]+val29);
  data0_196608[(alu28+2)] = (buf0[2]+val29);
  data0_196608[(alu28+2304)] = (buf0[9]+val30);
  data0_196608[(alu28+2305)] = (buf0[10]+val28);
  data0_196608[(alu28+2306)] = (buf0[11]+val29);
  data0_196608[(alu28+768)] = (buf0[3]+val30);
  data0_196608[(alu28+769)] = (buf0[4]+val28);
  data0_196608[(alu28+770)] = (buf0[5]+val29);
  data0_196608[alu28] = (buf0[0]+val30);
}`;

const r_8_16_8_4_4_4_dispatch_48 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[alu4];
    var val1 = data1_196608[(alu4+3072)];
    var val2 = data1_196608[(alu4+6144)];
    var val3 = data1_196608[(alu4+9216)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const r_8_16_8_4_4_4_dispatch_49 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[(alu4+256)];
    var val1 = data1_196608[(alu4+3328)];
    var val2 = data1_196608[(alu4+6400)];
    var val3 = data1_196608[(alu4+9472)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const E_2_32_8_16_2_4_dispatch_50 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+1)];
  var val1 = data1_196608[(alu0+32)];
  var val2 = data1_196608[(alu0+33)];
  var val3 = data1_196608[(alu0+64)];
  var val4 = data1_196608[(alu0+65)];
  var val5 = data1_196608[(alu0+96)];
  var val6 = data1_196608[(alu0+97)];
  var val7 = data1_196608[alu0];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val13);
  var alu10 = (val7*alu5*val12);
  var alu11 = (val1*alu6*val12);
  var alu12 = (val2*alu6*val13);
  var alu13 = (val3*alu7*val12);
  var alu14 = (val4*alu7*val13);
  var alu15 = (val5*alu8*val12);
  var alu16 = (val6*alu8*val13);
  data0_65536[alu4] = ((alu10*val14)-(alu9*val16));
  data0_65536[(alu4+1)] = ((alu9*val15)+(alu10*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const E_2_32_8_16_2_4_dispatch_51 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+256)];
  var val1 = data1_196608[(alu0+257)];
  var val2 = data1_196608[(alu0+288)];
  var val3 = data1_196608[(alu0+289)];
  var val4 = data1_196608[(alu0+320)];
  var val5 = data1_196608[(alu0+321)];
  var val6 = data1_196608[(alu0+352)];
  var val7 = data1_196608[(alu0+353)];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val12);
  var alu10 = (val1*alu5*val13);
  var alu11 = (val2*alu6*val12);
  var alu12 = (val3*alu6*val13);
  var alu13 = (val4*alu7*val12);
  var alu14 = (val5*alu7*val13);
  var alu15 = (val6*alu8*val12);
  var alu16 = (val7*alu8*val13);
  data0_65536[alu4] = ((alu9*val14)-(alu10*val16));
  data0_65536[(alu4+1)] = ((alu10*val15)+(alu9*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const r_8_8_4_8_16_4_4_32_dispatch_52 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_524288:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
  var cast2 = bitcast<i32>((cast1<<13u));
  var cast3 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<10u))+bitcast<i32>((cast3<<7u))+cast2);
  var val0 = data1_65536[alu0];
  var val1 = data1_65536[(alu0+1)];
  var val2 = data1_65536[(alu0+10)];
  var val3 = data1_65536[(alu0+100)];
  var val4 = data1_65536[(alu0+101)];
  var val5 = data1_65536[(alu0+102)];
  var val6 = data1_65536[(alu0+103)];
  var val7 = data1_65536[(alu0+104)];
  var val8 = data1_65536[(alu0+105)];
  var val9 = data1_65536[(alu0+106)];
  var val10 = data1_65536[(alu0+107)];
  var val11 = data1_65536[(alu0+108)];
  var val12 = data1_65536[(alu0+109)];
  var val13 = data1_65536[(alu0+11)];
  var val14 = data1_65536[(alu0+110)];
  var val15 = data1_65536[(alu0+111)];
  var val16 = data1_65536[(alu0+112)];
  var val17 = data1_65536[(alu0+113)];
  var val18 = data1_65536[(alu0+114)];
  var val19 = data1_65536[(alu0+115)];
  var val20 = data1_65536[(alu0+116)];
  var val21 = data1_65536[(alu0+117)];
  var val22 = data1_65536[(alu0+118)];
  var val23 = data1_65536[(alu0+119)];
  var val24 = data1_65536[(alu0+12)];
  var val25 = data1_65536[(alu0+120)];
  var val26 = data1_65536[(alu0+121)];
  var val27 = data1_65536[(alu0+122)];
  var val28 = data1_65536[(alu0+123)];
  var val29 = data1_65536[(alu0+124)];
  var val30 = data1_65536[(alu0+125)];
  var val31 = data1_65536[(alu0+126)];
  var val32 = data1_65536[(alu0+127)];
  var val33 = data1_65536[(alu0+13)];
  var val34 = data1_65536[(alu0+14)];
  var val35 = data1_65536[(alu0+15)];
  var val36 = data1_65536[(alu0+16)];
  var val37 = data1_65536[(alu0+17)];
  var val38 = data1_65536[(alu0+18)];
  var val39 = data1_65536[(alu0+19)];
  var val40 = data1_65536[(alu0+2)];
  var val41 = data1_65536[(alu0+20)];
  var val42 = data1_65536[(alu0+21)];
  var val43 = data1_65536[(alu0+22)];
  var val44 = data1_65536[(alu0+23)];
  var val45 = data1_65536[(alu0+24)];
  var val46 = data1_65536[(alu0+25)];
  var val47 = data1_65536[(alu0+26)];
  var val48 = data1_65536[(alu0+27)];
  var val49 = data1_65536[(alu0+28)];
  var val50 = data1_65536[(alu0+29)];
  var val51 = data1_65536[(alu0+3)];
  var val52 = data1_65536[(alu0+30)];
  var val53 = data1_65536[(alu0+31)];
  var val54 = data1_65536[(alu0+32)];
  var val55 = data1_65536[(alu0+33)];
  var val56 = data1_65536[(alu0+34)];
  var val57 = data1_65536[(alu0+35)];
  var val58 = data1_65536[(alu0+36)];
  var val59 = data1_65536[(alu0+37)];
  var val60 = data1_65536[(alu0+38)];
  var val61 = data1_65536[(alu0+39)];
  var val62 = data1_65536[(alu0+4)];
  var val63 = data1_65536[(alu0+40)];
  var val64 = data1_65536[(alu0+41)];
  var val65 = data1_65536[(alu0+42)];
  var val66 = data1_65536[(alu0+43)];
  var val67 = data1_65536[(alu0+44)];
  var val68 = data1_65536[(alu0+45)];
  var val69 = data1_65536[(alu0+46)];
  var val70 = data1_65536[(alu0+47)];
  var val71 = data1_65536[(alu0+48)];
  var val72 = data1_65536[(alu0+49)];
  var val73 = data1_65536[(alu0+5)];
  var val74 = data1_65536[(alu0+50)];
  var val75 = data1_65536[(alu0+51)];
  var val76 = data1_65536[(alu0+52)];
  var val77 = data1_65536[(alu0+53)];
  var val78 = data1_65536[(alu0+54)];
  var val79 = data1_65536[(alu0+55)];
  var val80 = data1_65536[(alu0+56)];
  var val81 = data1_65536[(alu0+57)];
  var val82 = data1_65536[(alu0+58)];
  var val83 = data1_65536[(alu0+59)];
  var val84 = data1_65536[(alu0+6)];
  var val85 = data1_65536[(alu0+60)];
  var val86 = data1_65536[(alu0+61)];
  var val87 = data1_65536[(alu0+62)];
  var val88 = data1_65536[(alu0+63)];
  var val89 = data1_65536[(alu0+64)];
  var val90 = data1_65536[(alu0+65)];
  var val91 = data1_65536[(alu0+66)];
  var val92 = data1_65536[(alu0+67)];
  var val93 = data1_65536[(alu0+68)];
  var val94 = data1_65536[(alu0+69)];
  var val95 = data1_65536[(alu0+7)];
  var val96 = data1_65536[(alu0+70)];
  var val97 = data1_65536[(alu0+71)];
  var val98 = data1_65536[(alu0+72)];
  var val99 = data1_65536[(alu0+73)];
  var val100 = data1_65536[(alu0+74)];
  var val101 = data1_65536[(alu0+75)];
  var val102 = data1_65536[(alu0+76)];
  var val103 = data1_65536[(alu0+77)];
  var val104 = data1_65536[(alu0+78)];
  var val105 = data1_65536[(alu0+79)];
  var val106 = data1_65536[(alu0+8)];
  var val107 = data1_65536[(alu0+80)];
  var val108 = data1_65536[(alu0+81)];
  var val109 = data1_65536[(alu0+82)];
  var val110 = data1_65536[(alu0+83)];
  var val111 = data1_65536[(alu0+84)];
  var val112 = data1_65536[(alu0+85)];
  var val113 = data1_65536[(alu0+86)];
  var val114 = data1_65536[(alu0+87)];
  var val115 = data1_65536[(alu0+88)];
  var val116 = data1_65536[(alu0+89)];
  var val117 = data1_65536[(alu0+9)];
  var val118 = data1_65536[(alu0+90)];
  var val119 = data1_65536[(alu0+91)];
  var val120 = data1_65536[(alu0+92)];
  var val121 = data1_65536[(alu0+93)];
  var val122 = data1_65536[(alu0+94)];
  var val123 = data1_65536[(alu0+95)];
  var val124 = data1_65536[(alu0+96)];
  var val125 = data1_65536[(alu0+97)];
  var val126 = data1_65536[(alu0+98)];
  var val127 = data1_65536[(alu0+99)];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast4 = bitcast<u32>(gidx0);
  var cast5 = bitcast<u32>(lidx1);
  var alu1 = (bitcast<i32>((cast4<<11u))+bitcast<i32>((cast5<<7u))+cast2);
  var val128 = data2_65536[alu1];
  var val129 = data2_65536[(alu1+1)];
  var val130 = data2_65536[(alu1+10)];
  var val131 = data2_65536[(alu1+100)];
  var val132 = data2_65536[(alu1+101)];
  var val133 = data2_65536[(alu1+102)];
  var val134 = data2_65536[(alu1+103)];
  var val135 = data2_65536[(alu1+104)];
  var val136 = data2_65536[(alu1+105)];
  var val137 = data2_65536[(alu1+106)];
  var val138 = data2_65536[(alu1+107)];
  var val139 = data2_65536[(alu1+108)];
  var val140 = data2_65536[(alu1+109)];
  var val141 = data2_65536[(alu1+11)];
  var val142 = data2_65536[(alu1+110)];
  var val143 = data2_65536[(alu1+111)];
  var val144 = data2_65536[(alu1+112)];
  var val145 = data2_65536[(alu1+113)];
  var val146 = data2_65536[(alu1+114)];
  var val147 = data2_65536[(alu1+115)];
  var val148 = data2_65536[(alu1+116)];
  var val149 = data2_65536[(alu1+117)];
  var val150 = data2_65536[(alu1+118)];
  var val151 = data2_65536[(alu1+119)];
  var val152 = data2_65536[(alu1+12)];
  var val153 = data2_65536[(alu1+120)];
  var val154 = data2_65536[(alu1+121)];
  var val155 = data2_65536[(alu1+122)];
  var val156 = data2_65536[(alu1+123)];
  var val157 = data2_65536[(alu1+124)];
  var val158 = data2_65536[(alu1+125)];
  var val159 = data2_65536[(alu1+126)];
  var val160 = data2_65536[(alu1+127)];
  var val161 = data2_65536[(alu1+13)];
  var val162 = data2_65536[(alu1+14)];
  var val163 = data2_65536[(alu1+15)];
  var val164 = data2_65536[(alu1+16)];
  var val165 = data2_65536[(alu1+17)];
  var val166 = data2_65536[(alu1+18)];
  var val167 = data2_65536[(alu1+19)];
  var val168 = data2_65536[(alu1+2)];
  var val169 = data2_65536[(alu1+20)];
  var val170 = data2_65536[(alu1+21)];
  var val171 = data2_65536[(alu1+22)];
  var val172 = data2_65536[(alu1+23)];
  var val173 = data2_65536[(alu1+24)];
  var val174 = data2_65536[(alu1+25)];
  var val175 = data2_65536[(alu1+26)];
  var val176 = data2_65536[(alu1+27)];
  var val177 = data2_65536[(alu1+28)];
  var val178 = data2_65536[(alu1+29)];
  var val179 = data2_65536[(alu1+3)];
  var val180 = data2_65536[(alu1+30)];
  var val181 = data2_65536[(alu1+31)];
  var val182 = data2_65536[(alu1+32)];
  var val183 = data2_65536[(alu1+33)];
  var val184 = data2_65536[(alu1+34)];
  var val185 = data2_65536[(alu1+35)];
  var val186 = data2_65536[(alu1+36)];
  var val187 = data2_65536[(alu1+37)];
  var val188 = data2_65536[(alu1+38)];
  var val189 = data2_65536[(alu1+39)];
  var val190 = data2_65536[(alu1+4)];
  var val191 = data2_65536[(alu1+40)];
  var val192 = data2_65536[(alu1+41)];
  var val193 = data2_65536[(alu1+42)];
  var val194 = data2_65536[(alu1+43)];
  var val195 = data2_65536[(alu1+44)];
  var val196 = data2_65536[(alu1+45)];
  var val197 = data2_65536[(alu1+46)];
  var val198 = data2_65536[(alu1+47)];
  var val199 = data2_65536[(alu1+48)];
  var val200 = data2_65536[(alu1+49)];
  var val201 = data2_65536[(alu1+5)];
  var val202 = data2_65536[(alu1+50)];
  var val203 = data2_65536[(alu1+51)];
  var val204 = data2_65536[(alu1+52)];
  var val205 = data2_65536[(alu1+53)];
  var val206 = data2_65536[(alu1+54)];
  var val207 = data2_65536[(alu1+55)];
  var val208 = data2_65536[(alu1+56)];
  var val209 = data2_65536[(alu1+57)];
  var val210 = data2_65536[(alu1+58)];
  var val211 = data2_65536[(alu1+59)];
  var val212 = data2_65536[(alu1+6)];
  var val213 = data2_65536[(alu1+60)];
  var val214 = data2_65536[(alu1+61)];
  var val215 = data2_65536[(alu1+62)];
  var val216 = data2_65536[(alu1+63)];
  var val217 = data2_65536[(alu1+64)];
  var val218 = data2_65536[(alu1+65)];
  var val219 = data2_65536[(alu1+66)];
  var val220 = data2_65536[(alu1+67)];
  var val221 = data2_65536[(alu1+68)];
  var val222 = data2_65536[(alu1+69)];
  var val223 = data2_65536[(alu1+7)];
  var val224 = data2_65536[(alu1+70)];
  var val225 = data2_65536[(alu1+71)];
  var val226 = data2_65536[(alu1+72)];
  var val227 = data2_65536[(alu1+73)];
  var val228 = data2_65536[(alu1+74)];
  var val229 = data2_65536[(alu1+75)];
  var val230 = data2_65536[(alu1+76)];
  var val231 = data2_65536[(alu1+77)];
  var val232 = data2_65536[(alu1+78)];
  var val233 = data2_65536[(alu1+79)];
  var val234 = data2_65536[(alu1+8)];
  var val235 = data2_65536[(alu1+80)];
  var val236 = data2_65536[(alu1+81)];
  var val237 = data2_65536[(alu1+82)];
  var val238 = data2_65536[(alu1+83)];
  var val239 = data2_65536[(alu1+84)];
  var val240 = data2_65536[(alu1+85)];
  var val241 = data2_65536[(alu1+86)];
  var val242 = data2_65536[(alu1+87)];
  var val243 = data2_65536[(alu1+88)];
  var val244 = data2_65536[(alu1+89)];
  var val245 = data2_65536[(alu1+9)];
  var val246 = data2_65536[(alu1+90)];
  var val247 = data2_65536[(alu1+91)];
  var val248 = data2_65536[(alu1+92)];
  var val249 = data2_65536[(alu1+93)];
  var val250 = data2_65536[(alu1+94)];
  var val251 = data2_65536[(alu1+95)];
  var val252 = data2_65536[(alu1+96)];
  var val253 = data2_65536[(alu1+97)];
  var val254 = data2_65536[(alu1+98)];
  var val255 = data2_65536[(alu1+99)];
  var alu2 = (bitcast<i32>((cast4<<6u))+bitcast<i32>((cast5<<2u))+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast1<<16u)));
  data0_524288[alu2] = (((val0*val128)+(val1*val129)+(val40*val168)+(val51*val179)+(val62*val190)+(val73*val201)+(val84*val212)+(val95*val223)+(val106*val234)+(val117*val245)+(val2*val130)+(val13*val141)+(val24*val152)+(val33*val161)+(val34*val162)+(val35*val163)+(val36*val164)+(val37*val165)+(val38*val166)+(val39*val167)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val173)+(val46*val174)+(val47*val175)+(val48*val176)+(val49*val177)+(val50*val178)+(val52*val180)+(val53*val181))*0.1767766922712326f);
  data0_524288[(alu2+1)] = (((val0*val182)+(val1*val183)+(val40*val184)+(val51*val185)+(val62*val186)+(val73*val187)+(val84*val188)+(val95*val189)+(val106*val191)+(val117*val192)+(val2*val193)+(val13*val194)+(val24*val195)+(val33*val196)+(val34*val197)+(val35*val198)+(val36*val199)+(val37*val200)+(val38*val202)+(val39*val203)+(val41*val204)+(val42*val205)+(val43*val206)+(val44*val207)+(val45*val208)+(val46*val209)+(val47*val210)+(val48*val211)+(val49*val213)+(val50*val214)+(val52*val215)+(val53*val216))*0.1767766922712326f);
  data0_524288[(alu2+2)] = (((val0*val217)+(val1*val218)+(val40*val219)+(val51*val220)+(val62*val221)+(val73*val222)+(val84*val224)+(val95*val225)+(val106*val226)+(val117*val227)+(val2*val228)+(val13*val229)+(val24*val230)+(val33*val231)+(val34*val232)+(val35*val233)+(val36*val235)+(val37*val236)+(val38*val237)+(val39*val238)+(val41*val239)+(val42*val240)+(val43*val241)+(val44*val242)+(val45*val243)+(val46*val244)+(val47*val246)+(val48*val247)+(val49*val248)+(val50*val249)+(val52*val250)+(val53*val251))*0.1767766922712326f);
  data0_524288[(alu2+256)] = (((val54*val128)+(val55*val129)+(val56*val168)+(val57*val179)+(val58*val190)+(val59*val201)+(val60*val212)+(val61*val223)+(val63*val234)+(val64*val245)+(val65*val130)+(val66*val141)+(val67*val152)+(val68*val161)+(val69*val162)+(val70*val163)+(val71*val164)+(val72*val165)+(val74*val166)+(val75*val167)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val173)+(val81*val174)+(val82*val175)+(val83*val176)+(val85*val177)+(val86*val178)+(val87*val180)+(val88*val181))*0.1767766922712326f);
  data0_524288[(alu2+257)] = (((val54*val182)+(val55*val183)+(val56*val184)+(val57*val185)+(val58*val186)+(val59*val187)+(val60*val188)+(val61*val189)+(val63*val191)+(val64*val192)+(val65*val193)+(val66*val194)+(val67*val195)+(val68*val196)+(val69*val197)+(val70*val198)+(val71*val199)+(val72*val200)+(val74*val202)+(val75*val203)+(val76*val204)+(val77*val205)+(val78*val206)+(val79*val207)+(val80*val208)+(val81*val209)+(val82*val210)+(val83*val211)+(val85*val213)+(val86*val214)+(val87*val215)+(val88*val216))*0.1767766922712326f);
  data0_524288[(alu2+258)] = (((val54*val217)+(val55*val218)+(val56*val219)+(val57*val220)+(val58*val221)+(val59*val222)+(val60*val224)+(val61*val225)+(val63*val226)+(val64*val227)+(val65*val228)+(val66*val229)+(val67*val230)+(val68*val231)+(val69*val232)+(val70*val233)+(val71*val235)+(val72*val236)+(val74*val237)+(val75*val238)+(val76*val239)+(val77*val240)+(val78*val241)+(val79*val242)+(val80*val243)+(val81*val244)+(val82*val246)+(val83*val247)+(val85*val248)+(val86*val249)+(val87*val250)+(val88*val251))*0.1767766922712326f);
  data0_524288[(alu2+259)] = (((val54*val252)+(val55*val253)+(val56*val254)+(val57*val255)+(val58*val131)+(val59*val132)+(val60*val133)+(val61*val134)+(val63*val135)+(val64*val136)+(val65*val137)+(val66*val138)+(val67*val139)+(val68*val140)+(val69*val142)+(val70*val143)+(val71*val144)+(val72*val145)+(val74*val146)+(val75*val147)+(val76*val148)+(val77*val149)+(val78*val150)+(val79*val151)+(val80*val153)+(val81*val154)+(val82*val155)+(val83*val156)+(val85*val157)+(val86*val158)+(val87*val159)+(val88*val160))*0.1767766922712326f);
  data0_524288[(alu2+3)] = (((val0*val252)+(val1*val253)+(val40*val254)+(val51*val255)+(val62*val131)+(val73*val132)+(val84*val133)+(val95*val134)+(val106*val135)+(val117*val136)+(val2*val137)+(val13*val138)+(val24*val139)+(val33*val140)+(val34*val142)+(val35*val143)+(val36*val144)+(val37*val145)+(val38*val146)+(val39*val147)+(val41*val148)+(val42*val149)+(val43*val150)+(val44*val151)+(val45*val153)+(val46*val154)+(val47*val155)+(val48*val156)+(val49*val157)+(val50*val158)+(val52*val159)+(val53*val160))*0.1767766922712326f);
  data0_524288[(alu2+512)] = (((val89*val128)+(val90*val129)+(val91*val168)+(val92*val179)+(val93*val190)+(val94*val201)+(val96*val212)+(val97*val223)+(val98*val234)+(val99*val245)+(val100*val130)+(val101*val141)+(val102*val152)+(val103*val161)+(val104*val162)+(val105*val163)+(val107*val164)+(val108*val165)+(val109*val166)+(val110*val167)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val173)+(val116*val174)+(val118*val175)+(val119*val176)+(val120*val177)+(val121*val178)+(val122*val180)+(val123*val181))*0.1767766922712326f);
  data0_524288[(alu2+513)] = (((val89*val182)+(val90*val183)+(val91*val184)+(val92*val185)+(val93*val186)+(val94*val187)+(val96*val188)+(val97*val189)+(val98*val191)+(val99*val192)+(val100*val193)+(val101*val194)+(val102*val195)+(val103*val196)+(val104*val197)+(val105*val198)+(val107*val199)+(val108*val200)+(val109*val202)+(val110*val203)+(val111*val204)+(val112*val205)+(val113*val206)+(val114*val207)+(val115*val208)+(val116*val209)+(val118*val210)+(val119*val211)+(val120*val213)+(val121*val214)+(val122*val215)+(val123*val216))*0.1767766922712326f);
  data0_524288[(alu2+514)] = (((val89*val217)+(val90*val218)+(val91*val219)+(val92*val220)+(val93*val221)+(val94*val222)+(val96*val224)+(val97*val225)+(val98*val226)+(val99*val227)+(val100*val228)+(val101*val229)+(val102*val230)+(val103*val231)+(val104*val232)+(val105*val233)+(val107*val235)+(val108*val236)+(val109*val237)+(val110*val238)+(val111*val239)+(val112*val240)+(val113*val241)+(val114*val242)+(val115*val243)+(val116*val244)+(val118*val246)+(val119*val247)+(val120*val248)+(val121*val249)+(val122*val250)+(val123*val251))*0.1767766922712326f);
  data0_524288[(alu2+515)] = (((val89*val252)+(val90*val253)+(val91*val254)+(val92*val255)+(val93*val131)+(val94*val132)+(val96*val133)+(val97*val134)+(val98*val135)+(val99*val136)+(val100*val137)+(val101*val138)+(val102*val139)+(val103*val140)+(val104*val142)+(val105*val143)+(val107*val144)+(val108*val145)+(val109*val146)+(val110*val147)+(val111*val148)+(val112*val149)+(val113*val150)+(val114*val151)+(val115*val153)+(val116*val154)+(val118*val155)+(val119*val156)+(val120*val157)+(val121*val158)+(val122*val159)+(val123*val160))*0.1767766922712326f);
  data0_524288[(alu2+768)] = (((val124*val128)+(val125*val129)+(val126*val168)+(val127*val179)+(val3*val190)+(val4*val201)+(val5*val212)+(val6*val223)+(val7*val234)+(val8*val245)+(val9*val130)+(val10*val141)+(val11*val152)+(val12*val161)+(val14*val162)+(val15*val163)+(val16*val164)+(val17*val165)+(val18*val166)+(val19*val167)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val173)+(val26*val174)+(val27*val175)+(val28*val176)+(val29*val177)+(val30*val178)+(val31*val180)+(val32*val181))*0.1767766922712326f);
  data0_524288[(alu2+769)] = (((val124*val182)+(val125*val183)+(val126*val184)+(val127*val185)+(val3*val186)+(val4*val187)+(val5*val188)+(val6*val189)+(val7*val191)+(val8*val192)+(val9*val193)+(val10*val194)+(val11*val195)+(val12*val196)+(val14*val197)+(val15*val198)+(val16*val199)+(val17*val200)+(val18*val202)+(val19*val203)+(val20*val204)+(val21*val205)+(val22*val206)+(val23*val207)+(val25*val208)+(val26*val209)+(val27*val210)+(val28*val211)+(val29*val213)+(val30*val214)+(val31*val215)+(val32*val216))*0.1767766922712326f);
  data0_524288[(alu2+770)] = (((val124*val217)+(val125*val218)+(val126*val219)+(val127*val220)+(val3*val221)+(val4*val222)+(val5*val224)+(val6*val225)+(val7*val226)+(val8*val227)+(val9*val228)+(val10*val229)+(val11*val230)+(val12*val231)+(val14*val232)+(val15*val233)+(val16*val235)+(val17*val236)+(val18*val237)+(val19*val238)+(val20*val239)+(val21*val240)+(val22*val241)+(val23*val242)+(val25*val243)+(val26*val244)+(val27*val246)+(val28*val247)+(val29*val248)+(val30*val249)+(val31*val250)+(val32*val251))*0.1767766922712326f);
  data0_524288[(alu2+771)] = (((val124*val252)+(val125*val253)+(val126*val254)+(val127*val255)+(val3*val131)+(val4*val132)+(val5*val133)+(val6*val134)+(val7*val135)+(val8*val136)+(val9*val137)+(val10*val138)+(val11*val139)+(val12*val140)+(val14*val142)+(val15*val143)+(val16*val144)+(val17*val145)+(val18*val146)+(val19*val147)+(val20*val148)+(val21*val149)+(val22*val150)+(val23*val151)+(val25*val153)+(val26*val154)+(val27*val155)+(val28*val156)+(val29*val157)+(val30*val158)+(val31*val159)+(val32*val160))*0.1767766922712326f);
}`;

const r_2048_16_16_dispatch_53 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = (f32(-INFINITY));
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val0 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    var val1 = buf0[0];
    var alu1 = select(val1,val0,(val1<val0));
    buf0[0] = alu1;
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = (f32(-INFINITY));
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf1[0];
    var val3 = buf2[Lidx102];
    var alu7 = select(val2,val3,(val2<val3));
    buf1[0] = alu7;
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_2048_16_16_dispatch_54 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var val0 = data2_2048[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val1 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    buf0[0] = (buf0[0]+exp2(((val1-val0)*1.4426950216293335f)));
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf2[Lidx102];
    buf1[0] = (buf1[0]+val2);
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_8_4_16_8_4_4_64_4_dispatch_55 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2048:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_196608:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u))+bitcast<i32>((cast1<<8u)));
  var val0 = data2_2048[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_2048[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_2048[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_2048[alu3];
  var lidx1 = i32(lindex.y); /* 8 */
  var cast3 = bitcast<i32>((bitcast<u32>(lidx1)<<2u));
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var alu20 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast2<<10u))+bitcast<i32>((bitcast<u32>(Lidx0)<<2u))+bitcast<i32>((cast1<<16u)));
    var val4 = data1_524288[alu20];
    var val5 = data1_524288[(alu20+1)];
    var val6 = data1_524288[(alu20+2)];
    var val7 = data1_524288[(alu20+256)];
    var val8 = data1_524288[(alu20+257)];
    var val9 = data1_524288[(alu20+258)];
    var val10 = data1_524288[(alu20+259)];
    var val11 = data1_524288[(alu20+3)];
    var val12 = data1_524288[(alu20+512)];
    var val13 = data1_524288[(alu20+513)];
    var val14 = data1_524288[(alu20+514)];
    var val15 = data1_524288[(alu20+515)];
    var val16 = data1_524288[(alu20+768)];
    var val17 = data1_524288[(alu20+769)];
    var val18 = data1_524288[(alu20+770)];
    var val19 = data1_524288[(alu20+771)];
    var alu21 = (bitcast<i32>((cast1<<5u))+cast3+(Lidx0*3072));
    var val20 = data4_196608[(alu21+1280)];
    var val21 = data4_196608[(alu21+1281)];
    var val22 = data4_196608[(alu21+1282)];
    var val23 = data4_196608[(alu21+1283)];
    var val24 = data4_196608[(alu21+2048)];
    var val25 = data4_196608[(alu21+2049)];
    var val26 = data4_196608[(alu21+2050)];
    var val27 = data4_196608[(alu21+2051)];
    var val28 = data4_196608[(alu21+2816)];
    var val29 = data4_196608[(alu21+2817)];
    var val30 = data4_196608[(alu21+2818)];
    var val31 = data4_196608[(alu21+2819)];
    var val32 = data4_196608[(alu21+512)];
    var val33 = data4_196608[(alu21+513)];
    var val34 = data4_196608[(alu21+514)];
    var val35 = data4_196608[(alu21+515)];
    var alu22 = exp2(((val4-val0)*1.4426950216293335f));
    var alu23 = exp2(((val5-val0)*1.4426950216293335f));
    var alu24 = exp2(((val6-val0)*1.4426950216293335f));
    var alu25 = exp2(((val7-val1)*1.4426950216293335f));
    var alu26 = exp2(((val8-val1)*1.4426950216293335f));
    var alu27 = exp2(((val9-val1)*1.4426950216293335f));
    var alu28 = exp2(((val10-val1)*1.4426950216293335f));
    var alu29 = exp2(((val11-val0)*1.4426950216293335f));
    var alu30 = exp2(((val12-val2)*1.4426950216293335f));
    var alu31 = exp2(((val13-val2)*1.4426950216293335f));
    var alu32 = exp2(((val14-val2)*1.4426950216293335f));
    var alu33 = exp2(((val15-val2)*1.4426950216293335f));
    var alu34 = exp2(((val16-val3)*1.4426950216293335f));
    var alu35 = exp2(((val17-val3)*1.4426950216293335f));
    var alu36 = exp2(((val18-val3)*1.4426950216293335f));
    var alu37 = exp2(((val19-val3)*1.4426950216293335f));
    buf0[0] = (buf0[0]+(alu22*val32)+(alu23*val20)+(alu24*val24)+(alu29*val28));
    buf0[1] = (buf0[1]+(alu22*val33)+(alu23*val21)+(alu24*val25)+(alu29*val29));
    buf0[10] = (buf0[10]+(alu30*val34)+(alu31*val22)+(alu32*val26)+(alu33*val30));
    buf0[11] = (buf0[11]+(alu30*val35)+(alu31*val23)+(alu32*val27)+(alu33*val31));
    buf0[12] = (buf0[12]+(alu34*val32)+(alu35*val20)+(alu36*val24)+(alu37*val28));
    buf0[13] = (buf0[13]+(alu34*val33)+(alu35*val21)+(alu36*val25)+(alu37*val29));
    buf0[14] = (buf0[14]+(alu34*val34)+(alu35*val22)+(alu36*val26)+(alu37*val30));
    buf0[15] = (buf0[15]+(alu34*val35)+(alu35*val23)+(alu36*val27)+(alu37*val31));
    buf0[2] = (buf0[2]+(alu22*val34)+(alu23*val22)+(alu24*val26)+(alu29*val30));
    buf0[3] = (buf0[3]+(alu22*val35)+(alu23*val23)+(alu24*val27)+(alu29*val31));
    buf0[4] = (buf0[4]+(alu25*val32)+(alu26*val20)+(alu27*val24)+(alu28*val28));
    buf0[5] = (buf0[5]+(alu25*val33)+(alu26*val21)+(alu27*val25)+(alu28*val29));
    buf0[6] = (buf0[6]+(alu25*val34)+(alu26*val22)+(alu27*val26)+(alu28*val30));
    buf0[7] = (buf0[7]+(alu25*val35)+(alu26*val23)+(alu27*val27)+(alu28*val31));
    buf0[8] = (buf0[8]+(alu30*val32)+(alu31*val20)+(alu32*val24)+(alu33*val28));
    buf0[9] = (buf0[9]+(alu30*val33)+(alu31*val21)+(alu32*val25)+(alu33*val29));
  }
  var val36 = data3_2048[alu0];
  var val37 = data3_2048[alu1];
  var val38 = data3_2048[alu2];
  var val39 = data3_2048[alu3];
  var alu55 = (bitcast<i32>((cast0<<11u))+bitcast<i32>((cast2<<7u))+cast3+bitcast<i32>((cast1<<13u)));
  var alu56 = (1/val36);
  var alu57 = (1/val37);
  var alu58 = (1/val38);
  var alu59 = (1/val39);
  data0_65536[alu55] = (buf0[0]*alu56);
  data0_65536[(alu55+1)] = (buf0[1]*alu56);
  data0_65536[(alu55+2)] = (buf0[2]*alu56);
  data0_65536[(alu55+3)] = (buf0[3]*alu56);
  data0_65536[(alu55+32)] = (buf0[4]*alu57);
  data0_65536[(alu55+33)] = (buf0[5]*alu57);
  data0_65536[(alu55+34)] = (buf0[6]*alu57);
  data0_65536[(alu55+35)] = (buf0[7]*alu57);
  data0_65536[(alu55+64)] = (buf0[8]*alu58);
  data0_65536[(alu55+65)] = (buf0[9]*alu58);
  data0_65536[(alu55+66)] = (buf0[10]*alu58);
  data0_65536[(alu55+67)] = (buf0[11]*alu58);
  data0_65536[(alu55+96)] = (buf0[12]*alu59);
  data0_65536[(alu55+97)] = (buf0[13]*alu59);
  data0_65536[(alu55+98)] = (buf0[14]*alu59);
  data0_65536[(alu55+99)] = (buf0[15]*alu59);
}`;

const r_8_4_8_16_4_4_8_32_dispatch_56 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_65536:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0_0 = 0; Lidx0_0 < 8; Lidx0_0++) {
    var cast4 = bitcast<u32>(Lidx0_0);
    var alu16 = (bitcast<i32>((cast1<<10u))+bitcast<i32>((cast2<<7u))+bitcast<i32>((cast4<<13u)));
    var val0 = data3_65536[alu16];
    var val1 = data3_65536[(alu16+1)];
    var val2 = data3_65536[(alu16+10)];
    var val3 = data3_65536[(alu16+100)];
    var val4 = data3_65536[(alu16+101)];
    var val5 = data3_65536[(alu16+102)];
    var val6 = data3_65536[(alu16+103)];
    var val7 = data3_65536[(alu16+104)];
    var val8 = data3_65536[(alu16+105)];
    var val9 = data3_65536[(alu16+106)];
    var val10 = data3_65536[(alu16+107)];
    var val11 = data3_65536[(alu16+108)];
    var val12 = data3_65536[(alu16+109)];
    var val13 = data3_65536[(alu16+11)];
    var val14 = data3_65536[(alu16+110)];
    var val15 = data3_65536[(alu16+111)];
    var val16 = data3_65536[(alu16+112)];
    var val17 = data3_65536[(alu16+113)];
    var val18 = data3_65536[(alu16+114)];
    var val19 = data3_65536[(alu16+115)];
    var val20 = data3_65536[(alu16+116)];
    var val21 = data3_65536[(alu16+117)];
    var val22 = data3_65536[(alu16+118)];
    var val23 = data3_65536[(alu16+119)];
    var val24 = data3_65536[(alu16+12)];
    var val25 = data3_65536[(alu16+120)];
    var val26 = data3_65536[(alu16+121)];
    var val27 = data3_65536[(alu16+122)];
    var val28 = data3_65536[(alu16+123)];
    var val29 = data3_65536[(alu16+124)];
    var val30 = data3_65536[(alu16+125)];
    var val31 = data3_65536[(alu16+126)];
    var val32 = data3_65536[(alu16+127)];
    var val33 = data3_65536[(alu16+13)];
    var val34 = data3_65536[(alu16+14)];
    var val35 = data3_65536[(alu16+15)];
    var val36 = data3_65536[(alu16+16)];
    var val37 = data3_65536[(alu16+17)];
    var val38 = data3_65536[(alu16+18)];
    var val39 = data3_65536[(alu16+19)];
    var val40 = data3_65536[(alu16+2)];
    var val41 = data3_65536[(alu16+20)];
    var val42 = data3_65536[(alu16+21)];
    var val43 = data3_65536[(alu16+22)];
    var val44 = data3_65536[(alu16+23)];
    var val45 = data3_65536[(alu16+24)];
    var val46 = data3_65536[(alu16+25)];
    var val47 = data3_65536[(alu16+26)];
    var val48 = data3_65536[(alu16+27)];
    var val49 = data3_65536[(alu16+28)];
    var val50 = data3_65536[(alu16+29)];
    var val51 = data3_65536[(alu16+3)];
    var val52 = data3_65536[(alu16+30)];
    var val53 = data3_65536[(alu16+31)];
    var val54 = data3_65536[(alu16+32)];
    var val55 = data3_65536[(alu16+33)];
    var val56 = data3_65536[(alu16+34)];
    var val57 = data3_65536[(alu16+35)];
    var val58 = data3_65536[(alu16+36)];
    var val59 = data3_65536[(alu16+37)];
    var val60 = data3_65536[(alu16+38)];
    var val61 = data3_65536[(alu16+39)];
    var val62 = data3_65536[(alu16+4)];
    var val63 = data3_65536[(alu16+40)];
    var val64 = data3_65536[(alu16+41)];
    var val65 = data3_65536[(alu16+42)];
    var val66 = data3_65536[(alu16+43)];
    var val67 = data3_65536[(alu16+44)];
    var val68 = data3_65536[(alu16+45)];
    var val69 = data3_65536[(alu16+46)];
    var val70 = data3_65536[(alu16+47)];
    var val71 = data3_65536[(alu16+48)];
    var val72 = data3_65536[(alu16+49)];
    var val73 = data3_65536[(alu16+5)];
    var val74 = data3_65536[(alu16+50)];
    var val75 = data3_65536[(alu16+51)];
    var val76 = data3_65536[(alu16+52)];
    var val77 = data3_65536[(alu16+53)];
    var val78 = data3_65536[(alu16+54)];
    var val79 = data3_65536[(alu16+55)];
    var val80 = data3_65536[(alu16+56)];
    var val81 = data3_65536[(alu16+57)];
    var val82 = data3_65536[(alu16+58)];
    var val83 = data3_65536[(alu16+59)];
    var val84 = data3_65536[(alu16+6)];
    var val85 = data3_65536[(alu16+60)];
    var val86 = data3_65536[(alu16+61)];
    var val87 = data3_65536[(alu16+62)];
    var val88 = data3_65536[(alu16+63)];
    var val89 = data3_65536[(alu16+64)];
    var val90 = data3_65536[(alu16+65)];
    var val91 = data3_65536[(alu16+66)];
    var val92 = data3_65536[(alu16+67)];
    var val93 = data3_65536[(alu16+68)];
    var val94 = data3_65536[(alu16+69)];
    var val95 = data3_65536[(alu16+7)];
    var val96 = data3_65536[(alu16+70)];
    var val97 = data3_65536[(alu16+71)];
    var val98 = data3_65536[(alu16+72)];
    var val99 = data3_65536[(alu16+73)];
    var val100 = data3_65536[(alu16+74)];
    var val101 = data3_65536[(alu16+75)];
    var val102 = data3_65536[(alu16+76)];
    var val103 = data3_65536[(alu16+77)];
    var val104 = data3_65536[(alu16+78)];
    var val105 = data3_65536[(alu16+79)];
    var val106 = data3_65536[(alu16+8)];
    var val107 = data3_65536[(alu16+80)];
    var val108 = data3_65536[(alu16+81)];
    var val109 = data3_65536[(alu16+82)];
    var val110 = data3_65536[(alu16+83)];
    var val111 = data3_65536[(alu16+84)];
    var val112 = data3_65536[(alu16+85)];
    var val113 = data3_65536[(alu16+86)];
    var val114 = data3_65536[(alu16+87)];
    var val115 = data3_65536[(alu16+88)];
    var val116 = data3_65536[(alu16+89)];
    var val117 = data3_65536[(alu16+9)];
    var val118 = data3_65536[(alu16+90)];
    var val119 = data3_65536[(alu16+91)];
    var val120 = data3_65536[(alu16+92)];
    var val121 = data3_65536[(alu16+93)];
    var val122 = data3_65536[(alu16+94)];
    var val123 = data3_65536[(alu16+95)];
    var val124 = data3_65536[(alu16+96)];
    var val125 = data3_65536[(alu16+97)];
    var val126 = data3_65536[(alu16+98)];
    var val127 = data3_65536[(alu16+99)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast4<<5u)));
    var val128 = data4_65536[alu17];
    var val129 = data4_65536[(alu17+1)];
    var val130 = data4_65536[(alu17+10)];
    var val131 = data4_65536[(alu17+11)];
    var val132 = data4_65536[(alu17+12)];
    var val133 = data4_65536[(alu17+13)];
    var val134 = data4_65536[(alu17+14)];
    var val135 = data4_65536[(alu17+15)];
    var val136 = data4_65536[(alu17+16)];
    var val137 = data4_65536[(alu17+17)];
    var val138 = data4_65536[(alu17+18)];
    var val139 = data4_65536[(alu17+19)];
    var val140 = data4_65536[(alu17+2)];
    var val141 = data4_65536[(alu17+20)];
    var val142 = data4_65536[(alu17+21)];
    var val143 = data4_65536[(alu17+22)];
    var val144 = data4_65536[(alu17+23)];
    var val145 = data4_65536[(alu17+24)];
    var val146 = data4_65536[(alu17+25)];
    var val147 = data4_65536[(alu17+256)];
    var val148 = data4_65536[(alu17+257)];
    var val149 = data4_65536[(alu17+258)];
    var val150 = data4_65536[(alu17+259)];
    var val151 = data4_65536[(alu17+26)];
    var val152 = data4_65536[(alu17+260)];
    var val153 = data4_65536[(alu17+261)];
    var val154 = data4_65536[(alu17+262)];
    var val155 = data4_65536[(alu17+263)];
    var val156 = data4_65536[(alu17+264)];
    var val157 = data4_65536[(alu17+265)];
    var val158 = data4_65536[(alu17+266)];
    var val159 = data4_65536[(alu17+267)];
    var val160 = data4_65536[(alu17+268)];
    var val161 = data4_65536[(alu17+269)];
    var val162 = data4_65536[(alu17+27)];
    var val163 = data4_65536[(alu17+270)];
    var val164 = data4_65536[(alu17+271)];
    var val165 = data4_65536[(alu17+272)];
    var val166 = data4_65536[(alu17+273)];
    var val167 = data4_65536[(alu17+274)];
    var val168 = data4_65536[(alu17+275)];
    var val169 = data4_65536[(alu17+276)];
    var val170 = data4_65536[(alu17+277)];
    var val171 = data4_65536[(alu17+278)];
    var val172 = data4_65536[(alu17+279)];
    var val173 = data4_65536[(alu17+28)];
    var val174 = data4_65536[(alu17+280)];
    var val175 = data4_65536[(alu17+281)];
    var val176 = data4_65536[(alu17+282)];
    var val177 = data4_65536[(alu17+283)];
    var val178 = data4_65536[(alu17+284)];
    var val179 = data4_65536[(alu17+285)];
    var val180 = data4_65536[(alu17+286)];
    var val181 = data4_65536[(alu17+287)];
    var val182 = data4_65536[(alu17+29)];
    var val183 = data4_65536[(alu17+3)];
    var val184 = data4_65536[(alu17+30)];
    var val185 = data4_65536[(alu17+31)];
    var val186 = data4_65536[(alu17+4)];
    var val187 = data4_65536[(alu17+5)];
    var val188 = data4_65536[(alu17+512)];
    var val189 = data4_65536[(alu17+513)];
    var val190 = data4_65536[(alu17+514)];
    var val191 = data4_65536[(alu17+515)];
    var val192 = data4_65536[(alu17+516)];
    var val193 = data4_65536[(alu17+517)];
    var val194 = data4_65536[(alu17+518)];
    var val195 = data4_65536[(alu17+519)];
    var val196 = data4_65536[(alu17+520)];
    var val197 = data4_65536[(alu17+521)];
    var val198 = data4_65536[(alu17+522)];
    var val199 = data4_65536[(alu17+523)];
    var val200 = data4_65536[(alu17+524)];
    var val201 = data4_65536[(alu17+525)];
    var val202 = data4_65536[(alu17+526)];
    var val203 = data4_65536[(alu17+527)];
    var val204 = data4_65536[(alu17+528)];
    var val205 = data4_65536[(alu17+529)];
    var val206 = data4_65536[(alu17+530)];
    var val207 = data4_65536[(alu17+531)];
    var val208 = data4_65536[(alu17+532)];
    var val209 = data4_65536[(alu17+533)];
    var val210 = data4_65536[(alu17+534)];
    var val211 = data4_65536[(alu17+535)];
    var val212 = data4_65536[(alu17+536)];
    var val213 = data4_65536[(alu17+537)];
    var val214 = data4_65536[(alu17+538)];
    var val215 = data4_65536[(alu17+539)];
    var val216 = data4_65536[(alu17+540)];
    var val217 = data4_65536[(alu17+541)];
    var val218 = data4_65536[(alu17+542)];
    var val219 = data4_65536[(alu17+543)];
    var val220 = data4_65536[(alu17+6)];
    var val221 = data4_65536[(alu17+7)];
    var val222 = data4_65536[(alu17+768)];
    var val223 = data4_65536[(alu17+769)];
    var val224 = data4_65536[(alu17+770)];
    var val225 = data4_65536[(alu17+771)];
    var val226 = data4_65536[(alu17+772)];
    var val227 = data4_65536[(alu17+773)];
    var val228 = data4_65536[(alu17+774)];
    var val229 = data4_65536[(alu17+775)];
    var val230 = data4_65536[(alu17+776)];
    var val231 = data4_65536[(alu17+777)];
    var val232 = data4_65536[(alu17+778)];
    var val233 = data4_65536[(alu17+779)];
    var val234 = data4_65536[(alu17+780)];
    var val235 = data4_65536[(alu17+781)];
    var val236 = data4_65536[(alu17+782)];
    var val237 = data4_65536[(alu17+783)];
    var val238 = data4_65536[(alu17+784)];
    var val239 = data4_65536[(alu17+785)];
    var val240 = data4_65536[(alu17+786)];
    var val241 = data4_65536[(alu17+787)];
    var val242 = data4_65536[(alu17+788)];
    var val243 = data4_65536[(alu17+789)];
    var val244 = data4_65536[(alu17+790)];
    var val245 = data4_65536[(alu17+791)];
    var val246 = data4_65536[(alu17+792)];
    var val247 = data4_65536[(alu17+793)];
    var val248 = data4_65536[(alu17+794)];
    var val249 = data4_65536[(alu17+795)];
    var val250 = data4_65536[(alu17+796)];
    var val251 = data4_65536[(alu17+797)];
    var val252 = data4_65536[(alu17+798)];
    var val253 = data4_65536[(alu17+799)];
    var val254 = data4_65536[(alu17+8)];
    var val255 = data4_65536[(alu17+9)];
    buf0[0] = (buf0[0]+(val0*val128)+(val1*val129)+(val40*val140)+(val51*val183)+(val62*val186)+(val73*val187)+(val84*val220)+(val95*val221)+(val106*val254)+(val117*val255)+(val2*val130)+(val13*val131)+(val24*val132)+(val33*val133)+(val34*val134)+(val35*val135)+(val36*val136)+(val37*val137)+(val38*val138)+(val39*val139)+(val41*val141)+(val42*val142)+(val43*val143)+(val44*val144)+(val45*val145)+(val46*val146)+(val47*val151)+(val48*val162)+(val49*val173)+(val50*val182)+(val52*val184)+(val53*val185));
    buf0[1] = (buf0[1]+(val0*val147)+(val1*val148)+(val40*val149)+(val51*val150)+(val62*val152)+(val73*val153)+(val84*val154)+(val95*val155)+(val106*val156)+(val117*val157)+(val2*val158)+(val13*val159)+(val24*val160)+(val33*val161)+(val34*val163)+(val35*val164)+(val36*val165)+(val37*val166)+(val38*val167)+(val39*val168)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val174)+(val46*val175)+(val47*val176)+(val48*val177)+(val49*val178)+(val50*val179)+(val52*val180)+(val53*val181));
    buf0[10] = (buf0[10]+(val89*val188)+(val90*val189)+(val91*val190)+(val92*val191)+(val93*val192)+(val94*val193)+(val96*val194)+(val97*val195)+(val98*val196)+(val99*val197)+(val100*val198)+(val101*val199)+(val102*val200)+(val103*val201)+(val104*val202)+(val105*val203)+(val107*val204)+(val108*val205)+(val109*val206)+(val110*val207)+(val111*val208)+(val112*val209)+(val113*val210)+(val114*val211)+(val115*val212)+(val116*val213)+(val118*val214)+(val119*val215)+(val120*val216)+(val121*val217)+(val122*val218)+(val123*val219));
    buf0[11] = (buf0[11]+(val89*val222)+(val90*val223)+(val91*val224)+(val92*val225)+(val93*val226)+(val94*val227)+(val96*val228)+(val97*val229)+(val98*val230)+(val99*val231)+(val100*val232)+(val101*val233)+(val102*val234)+(val103*val235)+(val104*val236)+(val105*val237)+(val107*val238)+(val108*val239)+(val109*val240)+(val110*val241)+(val111*val242)+(val112*val243)+(val113*val244)+(val114*val245)+(val115*val246)+(val116*val247)+(val118*val248)+(val119*val249)+(val120*val250)+(val121*val251)+(val122*val252)+(val123*val253));
    buf0[12] = (buf0[12]+(val124*val128)+(val125*val129)+(val126*val140)+(val127*val183)+(val3*val186)+(val4*val187)+(val5*val220)+(val6*val221)+(val7*val254)+(val8*val255)+(val9*val130)+(val10*val131)+(val11*val132)+(val12*val133)+(val14*val134)+(val15*val135)+(val16*val136)+(val17*val137)+(val18*val138)+(val19*val139)+(val20*val141)+(val21*val142)+(val22*val143)+(val23*val144)+(val25*val145)+(val26*val146)+(val27*val151)+(val28*val162)+(val29*val173)+(val30*val182)+(val31*val184)+(val32*val185));
    buf0[13] = (buf0[13]+(val124*val147)+(val125*val148)+(val126*val149)+(val127*val150)+(val3*val152)+(val4*val153)+(val5*val154)+(val6*val155)+(val7*val156)+(val8*val157)+(val9*val158)+(val10*val159)+(val11*val160)+(val12*val161)+(val14*val163)+(val15*val164)+(val16*val165)+(val17*val166)+(val18*val167)+(val19*val168)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val174)+(val26*val175)+(val27*val176)+(val28*val177)+(val29*val178)+(val30*val179)+(val31*val180)+(val32*val181));
    buf0[14] = (buf0[14]+(val124*val188)+(val125*val189)+(val126*val190)+(val127*val191)+(val3*val192)+(val4*val193)+(val5*val194)+(val6*val195)+(val7*val196)+(val8*val197)+(val9*val198)+(val10*val199)+(val11*val200)+(val12*val201)+(val14*val202)+(val15*val203)+(val16*val204)+(val17*val205)+(val18*val206)+(val19*val207)+(val20*val208)+(val21*val209)+(val22*val210)+(val23*val211)+(val25*val212)+(val26*val213)+(val27*val214)+(val28*val215)+(val29*val216)+(val30*val217)+(val31*val218)+(val32*val219));
    buf0[15] = (buf0[15]+(val124*val222)+(val125*val223)+(val126*val224)+(val127*val225)+(val3*val226)+(val4*val227)+(val5*val228)+(val6*val229)+(val7*val230)+(val8*val231)+(val9*val232)+(val10*val233)+(val11*val234)+(val12*val235)+(val14*val236)+(val15*val237)+(val16*val238)+(val17*val239)+(val18*val240)+(val19*val241)+(val20*val242)+(val21*val243)+(val22*val244)+(val23*val245)+(val25*val246)+(val26*val247)+(val27*val248)+(val28*val249)+(val29*val250)+(val30*val251)+(val31*val252)+(val32*val253));
    buf0[2] = (buf0[2]+(val0*val188)+(val1*val189)+(val40*val190)+(val51*val191)+(val62*val192)+(val73*val193)+(val84*val194)+(val95*val195)+(val106*val196)+(val117*val197)+(val2*val198)+(val13*val199)+(val24*val200)+(val33*val201)+(val34*val202)+(val35*val203)+(val36*val204)+(val37*val205)+(val38*val206)+(val39*val207)+(val41*val208)+(val42*val209)+(val43*val210)+(val44*val211)+(val45*val212)+(val46*val213)+(val47*val214)+(val48*val215)+(val49*val216)+(val50*val217)+(val52*val218)+(val53*val219));
    buf0[3] = (buf0[3]+(val0*val222)+(val1*val223)+(val40*val224)+(val51*val225)+(val62*val226)+(val73*val227)+(val84*val228)+(val95*val229)+(val106*val230)+(val117*val231)+(val2*val232)+(val13*val233)+(val24*val234)+(val33*val235)+(val34*val236)+(val35*val237)+(val36*val238)+(val37*val239)+(val38*val240)+(val39*val241)+(val41*val242)+(val42*val243)+(val43*val244)+(val44*val245)+(val45*val246)+(val46*val247)+(val47*val248)+(val48*val249)+(val49*val250)+(val50*val251)+(val52*val252)+(val53*val253));
    buf0[4] = (buf0[4]+(val54*val128)+(val55*val129)+(val56*val140)+(val57*val183)+(val58*val186)+(val59*val187)+(val60*val220)+(val61*val221)+(val63*val254)+(val64*val255)+(val65*val130)+(val66*val131)+(val67*val132)+(val68*val133)+(val69*val134)+(val70*val135)+(val71*val136)+(val72*val137)+(val74*val138)+(val75*val139)+(val76*val141)+(val77*val142)+(val78*val143)+(val79*val144)+(val80*val145)+(val81*val146)+(val82*val151)+(val83*val162)+(val85*val173)+(val86*val182)+(val87*val184)+(val88*val185));
    buf0[5] = (buf0[5]+(val54*val147)+(val55*val148)+(val56*val149)+(val57*val150)+(val58*val152)+(val59*val153)+(val60*val154)+(val61*val155)+(val63*val156)+(val64*val157)+(val65*val158)+(val66*val159)+(val67*val160)+(val68*val161)+(val69*val163)+(val70*val164)+(val71*val165)+(val72*val166)+(val74*val167)+(val75*val168)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val174)+(val81*val175)+(val82*val176)+(val83*val177)+(val85*val178)+(val86*val179)+(val87*val180)+(val88*val181));
    buf0[6] = (buf0[6]+(val54*val188)+(val55*val189)+(val56*val190)+(val57*val191)+(val58*val192)+(val59*val193)+(val60*val194)+(val61*val195)+(val63*val196)+(val64*val197)+(val65*val198)+(val66*val199)+(val67*val200)+(val68*val201)+(val69*val202)+(val70*val203)+(val71*val204)+(val72*val205)+(val74*val206)+(val75*val207)+(val76*val208)+(val77*val209)+(val78*val210)+(val79*val211)+(val80*val212)+(val81*val213)+(val82*val214)+(val83*val215)+(val85*val216)+(val86*val217)+(val87*val218)+(val88*val219));
    buf0[7] = (buf0[7]+(val54*val222)+(val55*val223)+(val56*val224)+(val57*val225)+(val58*val226)+(val59*val227)+(val60*val228)+(val61*val229)+(val63*val230)+(val64*val231)+(val65*val232)+(val66*val233)+(val67*val234)+(val68*val235)+(val69*val236)+(val70*val237)+(val71*val238)+(val72*val239)+(val74*val240)+(val75*val241)+(val76*val242)+(val77*val243)+(val78*val244)+(val79*val245)+(val80*val246)+(val81*val247)+(val82*val248)+(val83*val249)+(val85*val250)+(val86*val251)+(val87*val252)+(val88*val253));
    buf0[8] = (buf0[8]+(val89*val128)+(val90*val129)+(val91*val140)+(val92*val183)+(val93*val186)+(val94*val187)+(val96*val220)+(val97*val221)+(val98*val254)+(val99*val255)+(val100*val130)+(val101*val131)+(val102*val132)+(val103*val133)+(val104*val134)+(val105*val135)+(val107*val136)+(val108*val137)+(val109*val138)+(val110*val139)+(val111*val141)+(val112*val142)+(val113*val143)+(val114*val144)+(val115*val145)+(val116*val146)+(val118*val151)+(val119*val162)+(val120*val173)+(val121*val182)+(val122*val184)+(val123*val185));
    buf0[9] = (buf0[9]+(val89*val147)+(val90*val148)+(val91*val149)+(val92*val150)+(val93*val152)+(val94*val153)+(val96*val154)+(val97*val155)+(val98*val156)+(val99*val157)+(val100*val158)+(val101*val159)+(val102*val160)+(val103*val161)+(val104*val163)+(val105*val164)+(val107*val165)+(val108*val166)+(val109*val167)+(val110*val168)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val174)+(val116*val175)+(val118*val176)+(val119*val177)+(val120*val178)+(val121*val179)+(val122*val180)+(val123*val181));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<13u))+bitcast<i32>((cast2<<10u)));
  var val256 = data1_65536[alu36];
  var alu37 = (alu36+1);
  var val257 = data1_65536[alu37];
  var alu38 = (alu36+2);
  var val258 = data1_65536[alu38];
  var alu39 = (alu36+256);
  var val259 = data1_65536[alu39];
  var alu40 = (alu36+257);
  var val260 = data1_65536[alu40];
  var alu41 = (alu36+258);
  var val261 = data1_65536[alu41];
  var alu42 = (alu36+259);
  var val262 = data1_65536[alu42];
  var alu43 = (alu36+3);
  var val263 = data1_65536[alu43];
  var alu44 = (alu36+512);
  var val264 = data1_65536[alu44];
  var alu45 = (alu36+513);
  var val265 = data1_65536[alu45];
  var alu46 = (alu36+514);
  var val266 = data1_65536[alu46];
  var alu47 = (alu36+515);
  var val267 = data1_65536[alu47];
  var alu48 = (alu36+768);
  var val268 = data1_65536[alu48];
  var alu49 = (alu36+769);
  var val269 = data1_65536[alu49];
  var alu50 = (alu36+770);
  var val270 = data1_65536[alu50];
  var alu51 = (alu36+771);
  var val271 = data1_65536[alu51];
  var val272 = data2_1536[(alu35+512)];
  var val273 = data2_1536[(alu35+513)];
  var val274 = data2_1536[(alu35+514)];
  var val275 = data2_1536[(alu35+515)];
  var val276 = data5_256[alu35];
  var val277 = data5_256[(alu35+1)];
  var val278 = data5_256[(alu35+2)];
  var val279 = data5_256[(alu35+3)];
  data0_65536[alu36] = (val256+(val272*(buf0[0]+val276)));
  data0_65536[alu37] = (val257+(val273*(buf0[1]+val277)));
  data0_65536[alu38] = (val258+(val274*(buf0[2]+val278)));
  data0_65536[alu39] = (val259+(val272*(buf0[4]+val276)));
  data0_65536[alu40] = (val260+(val273*(buf0[5]+val277)));
  data0_65536[alu41] = (val261+(val274*(buf0[6]+val278)));
  data0_65536[alu42] = (val262+(val275*(buf0[7]+val279)));
  data0_65536[alu43] = (val263+(val275*(buf0[3]+val279)));
  data0_65536[alu44] = (val264+(val272*(buf0[8]+val276)));
  data0_65536[alu45] = (val265+(val273*(buf0[9]+val277)));
  data0_65536[alu46] = (val266+(val274*(buf0[10]+val278)));
  data0_65536[alu47] = (val267+(val275*(buf0[11]+val279)));
  data0_65536[alu48] = (val268+(val272*(buf0[12]+val276)));
  data0_65536[alu49] = (val269+(val273*(buf0[13]+val277)));
  data0_65536[alu50] = (val270+(val274*(buf0[14]+val278)));
  data0_65536[alu51] = (val271+(val275*(buf0[15]+val279)));
}`;

const r_16_8_4_4_32_dispatch_57 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_58 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var val21 = data3_256[(alu0+1)];
  var val22 = data3_256[(alu0+2)];
  var val23 = data3_256[(alu0+3)];
  var val24 = data4_1536[(alu0+1024)];
  var val25 = data4_1536[(alu0+1025)];
  var val26 = data4_1536[(alu0+1026)];
  var val27 = data4_1536[(alu0+1027)];
  var val28 = data4_1536[(alu0+768)];
  var val29 = data4_1536[(alu0+769)];
  var val30 = data4_1536[(alu0+770)];
  var val31 = data4_1536[(alu0+771)];
  var alu18 = (1/val16);
  var alu19 = (1/val17);
  var alu20 = (1/val18);
  var alu21 = (1/val19);
  var alu22 = (1.0f+val24);
  var alu23 = (1.0f+val25);
  var alu24 = (1.0f+val26);
  var alu25 = (1.0f+val27);
  data0_65536[alu1] = ((val0*alu18*val20*alu22)+val28);
  data0_65536[alu2] = ((val1*alu18*val21*alu23)+val29);
  data0_65536[alu3] = ((val2*alu18*val22*alu24)+val30);
  data0_65536[alu4] = ((val3*alu19*val20*alu22)+val28);
  data0_65536[alu5] = ((val4*alu19*val21*alu23)+val29);
  data0_65536[alu6] = ((val5*alu19*val22*alu24)+val30);
  data0_65536[alu7] = ((val6*alu19*val23*alu25)+val31);
  data0_65536[alu8] = ((val7*alu18*val23*alu25)+val31);
  data0_65536[alu9] = ((val8*alu20*val20*alu22)+val28);
  data0_65536[alu10] = ((val9*alu20*val21*alu23)+val29);
  data0_65536[alu11] = ((val10*alu20*val22*alu24)+val30);
  data0_65536[alu12] = ((val11*alu20*val23*alu25)+val31);
  data0_65536[alu13] = ((val12*alu21*val20*alu22)+val28);
  data0_65536[alu14] = ((val13*alu21*val21*alu23)+val29);
  data0_65536[alu15] = ((val14*alu21*val22*alu24)+val30);
  data0_65536[alu16] = ((val15*alu21*val23*alu25)+val31);
}`;

const r_8_22_8_16_4_4_64_4_dispatch_59 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_360448:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_360448:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1408:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 22 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast2 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast2);
    var val0 = data1_65536[alu16];
    var val1 = data1_65536[(alu16+1)];
    var val2 = data1_65536[(alu16+2)];
    var val3 = data1_65536[(alu16+256)];
    var val4 = data1_65536[(alu16+257)];
    var val5 = data1_65536[(alu16+258)];
    var val6 = data1_65536[(alu16+259)];
    var val7 = data1_65536[(alu16+3)];
    var val8 = data1_65536[(alu16+512)];
    var val9 = data1_65536[(alu16+513)];
    var val10 = data1_65536[(alu16+514)];
    var val11 = data1_65536[(alu16+515)];
    var val12 = data1_65536[(alu16+768)];
    var val13 = data1_65536[(alu16+769)];
    var val14 = data1_65536[(alu16+770)];
    var val15 = data1_65536[(alu16+771)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast1<<10u))+cast2);
    var val16 = data2_360448[alu17];
    var val17 = data2_360448[(alu17+1)];
    var val18 = data2_360448[(alu17+2)];
    var val19 = data2_360448[(alu17+256)];
    var val20 = data2_360448[(alu17+257)];
    var val21 = data2_360448[(alu17+258)];
    var val22 = data2_360448[(alu17+259)];
    var val23 = data2_360448[(alu17+3)];
    var val24 = data2_360448[(alu17+512)];
    var val25 = data2_360448[(alu17+513)];
    var val26 = data2_360448[(alu17+514)];
    var val27 = data2_360448[(alu17+515)];
    var val28 = data2_360448[(alu17+768)];
    var val29 = data2_360448[(alu17+769)];
    var val30 = data2_360448[(alu17+770)];
    var val31 = data2_360448[(alu17+771)];
    buf0[0] = (buf0[0]+(val0*val16)+(val1*val17)+(val2*val18)+(val7*val23));
    buf0[1] = (buf0[1]+(val0*val19)+(val1*val20)+(val2*val21)+(val7*val22));
    buf0[10] = (buf0[10]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    buf0[11] = (buf0[11]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    buf0[12] = (buf0[12]+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val23));
    buf0[13] = (buf0[13]+(val12*val19)+(val13*val20)+(val14*val21)+(val15*val22));
    buf0[14] = (buf0[14]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    buf0[15] = (buf0[15]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    buf0[2] = (buf0[2]+(val0*val24)+(val1*val25)+(val2*val26)+(val7*val27));
    buf0[3] = (buf0[3]+(val0*val28)+(val1*val29)+(val2*val30)+(val7*val31));
    buf0[4] = (buf0[4]+(val3*val16)+(val4*val17)+(val5*val18)+(val6*val23));
    buf0[5] = (buf0[5]+(val3*val19)+(val4*val20)+(val5*val21)+(val6*val22));
    buf0[6] = (buf0[6]+(val3*val24)+(val4*val25)+(val5*val26)+(val6*val27));
    buf0[7] = (buf0[7]+(val3*val28)+(val4*val29)+(val5*val30)+(val6*val31));
    buf0[8] = (buf0[8]+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val23));
    buf0[9] = (buf0[9]+(val8*val19)+(val9*val20)+(val10*val21)+(val11*val22));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)));
  var val32 = data3_1408[alu35];
  var val33 = data3_1408[(alu35+1)];
  var val34 = data3_1408[(alu35+2)];
  var val35 = data3_1408[(alu35+3)];
  var alu36 = (alu35+(gidx1*45056)+(lidx0*5632));
  data0_360448[alu36] = (buf0[0]+val32);
  data0_360448[(alu36+1)] = (buf0[1]+val33);
  data0_360448[(alu36+1408)] = (buf0[4]+val32);
  data0_360448[(alu36+1409)] = (buf0[5]+val33);
  data0_360448[(alu36+1410)] = (buf0[6]+val34);
  data0_360448[(alu36+1411)] = (buf0[7]+val35);
  data0_360448[(alu36+2)] = (buf0[2]+val34);
  data0_360448[(alu36+2816)] = (buf0[8]+val32);
  data0_360448[(alu36+2817)] = (buf0[9]+val33);
  data0_360448[(alu36+2818)] = (buf0[10]+val34);
  data0_360448[(alu36+2819)] = (buf0[11]+val35);
  data0_360448[(alu36+3)] = (buf0[3]+val35);
  data0_360448[(alu36+4224)] = (buf0[12]+val32);
  data0_360448[(alu36+4225)] = (buf0[13]+val33);
  data0_360448[(alu36+4226)] = (buf0[14]+val34);
  data0_360448[(alu36+4227)] = (buf0[15]+val35);
}`;

const r_8_4_8_16_4_4_176_4_dispatch_60 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_360448:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_180224:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 176; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = ((gidx1*45056)+(lidx0*5632)+cast0);
    var val0 = data3_360448[(alu16+1)];
    var val1 = data3_360448[(alu16+1408)];
    var val2 = data3_360448[(alu16+1409)];
    var val3 = data3_360448[(alu16+1410)];
    var val4 = data3_360448[(alu16+1411)];
    var val5 = data3_360448[(alu16+2)];
    var val6 = data3_360448[(alu16+2112)];
    var val7 = data3_360448[(alu16+2113)];
    var val8 = data3_360448[(alu16+2114)];
    var val9 = data3_360448[(alu16+2115)];
    var val10 = data3_360448[(alu16+2816)];
    var val11 = data3_360448[(alu16+2817)];
    var val12 = data3_360448[(alu16+2818)];
    var val13 = data3_360448[(alu16+2819)];
    var val14 = data3_360448[(alu16+3)];
    var val15 = data3_360448[(alu16+3520)];
    var val16 = data3_360448[(alu16+3521)];
    var val17 = data3_360448[(alu16+3522)];
    var val18 = data3_360448[(alu16+3523)];
    var val19 = data3_360448[(alu16+4224)];
    var val20 = data3_360448[(alu16+4225)];
    var val21 = data3_360448[(alu16+4226)];
    var val22 = data3_360448[(alu16+4227)];
    var val23 = data3_360448[(alu16+4928)];
    var val24 = data3_360448[(alu16+4929)];
    var val25 = data3_360448[(alu16+4930)];
    var val26 = data3_360448[(alu16+4931)];
    var val27 = data3_360448[(alu16+704)];
    var val28 = data3_360448[(alu16+705)];
    var val29 = data3_360448[(alu16+706)];
    var val30 = data3_360448[(alu16+707)];
    var val31 = data3_360448[alu16];
    var alu17 = ((gidx0*45056)+(lidx1*2816)+cast0);
    var val32 = data4_180224[(alu17+1)];
    var val33 = data4_180224[(alu17+1408)];
    var val34 = data4_180224[(alu17+1409)];
    var val35 = data4_180224[(alu17+1410)];
    var val36 = data4_180224[(alu17+1411)];
    var val37 = data4_180224[(alu17+2)];
    var val38 = data4_180224[(alu17+2112)];
    var val39 = data4_180224[(alu17+2113)];
    var val40 = data4_180224[(alu17+2114)];
    var val41 = data4_180224[(alu17+2115)];
    var val42 = data4_180224[(alu17+3)];
    var val43 = data4_180224[(alu17+704)];
    var val44 = data4_180224[(alu17+705)];
    var val45 = data4_180224[(alu17+706)];
    var val46 = data4_180224[(alu17+707)];
    var val47 = data4_180224[alu17];
    var alu18 = (val0*(1/(1.0f+exp2((val0*-1.4426950216293335f))))*val28);
    var alu19 = (val31*(1/(1.0f+exp2((val31*-1.4426950216293335f))))*val27);
    var alu20 = (val5*(1/(1.0f+exp2((val5*-1.4426950216293335f))))*val29);
    var alu21 = (val14*(1/(1.0f+exp2((val14*-1.4426950216293335f))))*val30);
    var alu22 = (val10*(1/(1.0f+exp2((val10*-1.4426950216293335f))))*val15);
    var alu23 = (val11*(1/(1.0f+exp2((val11*-1.4426950216293335f))))*val16);
    var alu24 = (val12*(1/(1.0f+exp2((val12*-1.4426950216293335f))))*val17);
    var alu25 = (val13*(1/(1.0f+exp2((val13*-1.4426950216293335f))))*val18);
    var alu26 = (val19*(1/(1.0f+exp2((val19*-1.4426950216293335f))))*val23);
    var alu27 = (val20*(1/(1.0f+exp2((val20*-1.4426950216293335f))))*val24);
    var alu28 = (val21*(1/(1.0f+exp2((val21*-1.4426950216293335f))))*val25);
    var alu29 = (val22*(1/(1.0f+exp2((val22*-1.4426950216293335f))))*val26);
    var alu30 = (val1*(1/(1.0f+exp2((val1*-1.4426950216293335f))))*val6);
    var alu31 = (val2*(1/(1.0f+exp2((val2*-1.4426950216293335f))))*val7);
    var alu32 = (val3*(1/(1.0f+exp2((val3*-1.4426950216293335f))))*val8);
    var alu33 = (val4*(1/(1.0f+exp2((val4*-1.4426950216293335f))))*val9);
    buf0[0] = (buf0[0]+(alu19*val47)+(alu18*val32)+(alu20*val37)+(alu21*val42));
    buf0[1] = (buf0[1]+(alu19*val43)+(alu18*val44)+(alu20*val45)+(alu21*val46));
    buf0[10] = (buf0[10]+(alu22*val33)+(alu23*val34)+(alu24*val35)+(alu25*val36));
    buf0[11] = (buf0[11]+(alu22*val38)+(alu23*val39)+(alu24*val40)+(alu25*val41));
    buf0[12] = (buf0[12]+(alu26*val47)+(alu27*val32)+(alu28*val37)+(alu29*val42));
    buf0[13] = (buf0[13]+(alu26*val43)+(alu27*val44)+(alu28*val45)+(alu29*val46));
    buf0[14] = (buf0[14]+(alu26*val33)+(alu27*val34)+(alu28*val35)+(alu29*val36));
    buf0[15] = (buf0[15]+(alu26*val38)+(alu27*val39)+(alu28*val40)+(alu29*val41));
    buf0[2] = (buf0[2]+(alu19*val33)+(alu18*val34)+(alu20*val35)+(alu21*val36));
    buf0[3] = (buf0[3]+(alu19*val38)+(alu18*val39)+(alu20*val40)+(alu21*val41));
    buf0[4] = (buf0[4]+(alu30*val47)+(alu31*val32)+(alu32*val37)+(alu33*val42));
    buf0[5] = (buf0[5]+(alu30*val43)+(alu31*val44)+(alu32*val45)+(alu33*val46));
    buf0[6] = (buf0[6]+(alu30*val33)+(alu31*val34)+(alu32*val35)+(alu33*val36));
    buf0[7] = (buf0[7]+(alu30*val38)+(alu31*val39)+(alu32*val40)+(alu33*val41));
    buf0[8] = (buf0[8]+(alu22*val47)+(alu23*val32)+(alu24*val37)+(alu25*val42));
    buf0[9] = (buf0[9]+(alu22*val43)+(alu23*val44)+(alu24*val45)+(alu25*val46));
  }
  var alu51 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu52 = (alu51+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u)));
  var val48 = data1_65536[alu52];
  var alu53 = (alu52+1);
  var val49 = data1_65536[alu53];
  var alu54 = (alu52+2);
  var val50 = data1_65536[alu54];
  var alu55 = (alu52+256);
  var val51 = data1_65536[alu55];
  var alu56 = (alu52+257);
  var val52 = data1_65536[alu56];
  var alu57 = (alu52+258);
  var val53 = data1_65536[alu57];
  var alu58 = (alu52+259);
  var val54 = data1_65536[alu58];
  var alu59 = (alu52+3);
  var val55 = data1_65536[alu59];
  var alu60 = (alu52+512);
  var val56 = data1_65536[alu60];
  var alu61 = (alu52+513);
  var val57 = data1_65536[alu61];
  var alu62 = (alu52+514);
  var val58 = data1_65536[alu62];
  var alu63 = (alu52+515);
  var val59 = data1_65536[alu63];
  var alu64 = (alu52+768);
  var val60 = data1_65536[alu64];
  var alu65 = (alu52+769);
  var val61 = data1_65536[alu65];
  var alu66 = (alu52+770);
  var val62 = data1_65536[alu66];
  var alu67 = (alu52+771);
  var val63 = data1_65536[alu67];
  var val64 = data2_1536[(alu51+1280)];
  var val65 = data2_1536[(alu51+1281)];
  var val66 = data2_1536[(alu51+1282)];
  var val67 = data2_1536[(alu51+1283)];
  var val68 = data5_256[alu51];
  var val69 = data5_256[(alu51+1)];
  var val70 = data5_256[(alu51+2)];
  var val71 = data5_256[(alu51+3)];
  data0_65536[alu52] = (val48+(val64*(buf0[0]+val68)));
  data0_65536[alu53] = (val49+(val65*(buf0[1]+val69)));
  data0_65536[alu54] = (val50+(val66*(buf0[2]+val70)));
  data0_65536[alu55] = (val51+(val64*(buf0[4]+val68)));
  data0_65536[alu56] = (val52+(val65*(buf0[5]+val69)));
  data0_65536[alu57] = (val53+(val66*(buf0[6]+val70)));
  data0_65536[alu58] = (val54+(val67*(buf0[7]+val71)));
  data0_65536[alu59] = (val55+(val67*(buf0[3]+val71)));
  data0_65536[alu60] = (val56+(val64*(buf0[8]+val68)));
  data0_65536[alu61] = (val57+(val65*(buf0[9]+val69)));
  data0_65536[alu62] = (val58+(val66*(buf0[10]+val70)));
  data0_65536[alu63] = (val59+(val67*(buf0[11]+val71)));
  data0_65536[alu64] = (val60+(val64*(buf0[12]+val68)));
  data0_65536[alu65] = (val61+(val65*(buf0[13]+val69)));
  data0_65536[alu66] = (val62+(val66*(buf0[14]+val70)));
  data0_65536[alu67] = (val63+(val67*(buf0[15]+val71)));
}`;

const r_16_8_4_4_32_dispatch_61 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_62 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var alu18 = (alu0+1);
  var val21 = data3_256[alu18];
  var alu19 = (alu0+2);
  var val22 = data3_256[alu19];
  var alu20 = (alu0+3);
  var val23 = data3_256[alu20];
  var val24 = data4_1536[alu0];
  var val25 = data4_1536[alu18];
  var val26 = data4_1536[alu19];
  var val27 = data4_1536[(alu0+256)];
  var val28 = data4_1536[(alu0+257)];
  var val29 = data4_1536[(alu0+258)];
  var val30 = data4_1536[(alu0+259)];
  var val31 = data4_1536[alu20];
  var alu21 = (1/val16);
  var alu22 = (1/val17);
  var alu23 = (1/val18);
  var alu24 = (1/val19);
  var alu25 = (1.0f+val27);
  var alu26 = (1.0f+val28);
  var alu27 = (1.0f+val29);
  var alu28 = (1.0f+val30);
  data0_65536[alu1] = ((val0*alu21*val20*alu25)+val24);
  data0_65536[alu2] = ((val1*alu21*val21*alu26)+val25);
  data0_65536[alu3] = ((val2*alu21*val22*alu27)+val26);
  data0_65536[alu4] = ((val3*alu22*val20*alu25)+val24);
  data0_65536[alu5] = ((val4*alu22*val21*alu26)+val25);
  data0_65536[alu6] = ((val5*alu22*val22*alu27)+val26);
  data0_65536[alu7] = ((val6*alu22*val23*alu28)+val31);
  data0_65536[alu8] = ((val7*alu21*val23*alu28)+val31);
  data0_65536[alu9] = ((val8*alu23*val20*alu25)+val24);
  data0_65536[alu10] = ((val9*alu23*val21*alu26)+val25);
  data0_65536[alu11] = ((val10*alu23*val22*alu27)+val26);
  data0_65536[alu12] = ((val11*alu23*val23*alu28)+val31);
  data0_65536[alu13] = ((val12*alu24*val20*alu25)+val24);
  data0_65536[alu14] = ((val13*alu24*val21*alu26)+val25);
  data0_65536[alu15] = ((val14*alu24*val22*alu27)+val26);
  data0_65536[alu16] = ((val15*alu24*val23*alu28)+val31);
}`;

const r_8_16_8_16_3_4_64_4_dispatch_63 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_196608:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu12 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast0);
    var val0 = data1_65536[alu12];
    var val1 = data1_65536[(alu12+1)];
    var val2 = data1_65536[(alu12+2)];
    var val3 = data1_65536[(alu12+256)];
    var val4 = data1_65536[(alu12+257)];
    var val5 = data1_65536[(alu12+258)];
    var val6 = data1_65536[(alu12+259)];
    var val7 = data1_65536[(alu12+3)];
    var val8 = data1_65536[(alu12+512)];
    var val9 = data1_65536[(alu12+513)];
    var val10 = data1_65536[(alu12+514)];
    var val11 = data1_65536[(alu12+515)];
    var val12 = data1_65536[(alu12+768)];
    var val13 = data1_65536[(alu12+769)];
    var val14 = data1_65536[(alu12+770)];
    var val15 = data1_65536[(alu12+771)];
    var alu13 = ((gidx0*12288)+(lidx1*768)+cast0);
    var val16 = data2_196608[(alu13+1)];
    var val17 = data2_196608[(alu13+2)];
    var val18 = data2_196608[(alu13+256)];
    var val19 = data2_196608[(alu13+257)];
    var val20 = data2_196608[(alu13+258)];
    var val21 = data2_196608[(alu13+259)];
    var val22 = data2_196608[(alu13+3)];
    var val23 = data2_196608[(alu13+512)];
    var val24 = data2_196608[(alu13+513)];
    var val25 = data2_196608[(alu13+514)];
    var val26 = data2_196608[(alu13+515)];
    var val27 = data2_196608[alu13];
    buf0[0] = (buf0[0]+(val0*val27)+(val1*val16)+(val2*val17)+(val7*val22));
    buf0[1] = (buf0[1]+(val0*val18)+(val1*val19)+(val2*val20)+(val7*val21));
    buf0[10] = (buf0[10]+(val12*val18)+(val13*val19)+(val14*val20)+(val15*val21));
    buf0[11] = (buf0[11]+(val12*val23)+(val13*val24)+(val14*val25)+(val15*val26));
    buf0[2] = (buf0[2]+(val0*val23)+(val1*val24)+(val2*val25)+(val7*val26));
    buf0[3] = (buf0[3]+(val3*val27)+(val4*val16)+(val5*val17)+(val6*val22));
    buf0[4] = (buf0[4]+(val3*val18)+(val4*val19)+(val5*val20)+(val6*val21));
    buf0[5] = (buf0[5]+(val3*val23)+(val4*val24)+(val5*val25)+(val6*val26));
    buf0[6] = (buf0[6]+(val8*val27)+(val9*val16)+(val10*val17)+(val11*val22));
    buf0[7] = (buf0[7]+(val8*val18)+(val9*val19)+(val10*val20)+(val11*val21));
    buf0[8] = (buf0[8]+(val8*val23)+(val9*val24)+(val10*val25)+(val11*val26));
    buf0[9] = (buf0[9]+(val12*val27)+(val13*val16)+(val14*val17)+(val15*val22));
  }
  var alu27 = ((gidx0*48)+(lidx1*3));
  var val28 = data3_768[(alu27+1)];
  var val29 = data3_768[(alu27+2)];
  var val30 = data3_768[alu27];
  var alu28 = (alu27+(gidx1*24576)+(lidx0*3072));
  data0_196608[(alu28+1)] = (buf0[1]+val28);
  data0_196608[(alu28+1536)] = (buf0[6]+val30);
  data0_196608[(alu28+1537)] = (buf0[7]+val28);
  data0_196608[(alu28+1538)] = (buf0[8]+val29);
  data0_196608[(alu28+2)] = (buf0[2]+val29);
  data0_196608[(alu28+2304)] = (buf0[9]+val30);
  data0_196608[(alu28+2305)] = (buf0[10]+val28);
  data0_196608[(alu28+2306)] = (buf0[11]+val29);
  data0_196608[(alu28+768)] = (buf0[3]+val30);
  data0_196608[(alu28+769)] = (buf0[4]+val28);
  data0_196608[(alu28+770)] = (buf0[5]+val29);
  data0_196608[alu28] = (buf0[0]+val30);
}`;

const r_8_16_8_4_4_4_dispatch_64 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[alu4];
    var val1 = data1_196608[(alu4+3072)];
    var val2 = data1_196608[(alu4+6144)];
    var val3 = data1_196608[(alu4+9216)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const r_8_16_8_4_4_4_dispatch_65 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[(alu4+256)];
    var val1 = data1_196608[(alu4+3328)];
    var val2 = data1_196608[(alu4+6400)];
    var val3 = data1_196608[(alu4+9472)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const E_2_32_8_16_2_4_dispatch_66 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+1)];
  var val1 = data1_196608[(alu0+32)];
  var val2 = data1_196608[(alu0+33)];
  var val3 = data1_196608[(alu0+64)];
  var val4 = data1_196608[(alu0+65)];
  var val5 = data1_196608[(alu0+96)];
  var val6 = data1_196608[(alu0+97)];
  var val7 = data1_196608[alu0];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val13);
  var alu10 = (val7*alu5*val12);
  var alu11 = (val1*alu6*val12);
  var alu12 = (val2*alu6*val13);
  var alu13 = (val3*alu7*val12);
  var alu14 = (val4*alu7*val13);
  var alu15 = (val5*alu8*val12);
  var alu16 = (val6*alu8*val13);
  data0_65536[alu4] = ((alu10*val14)-(alu9*val16));
  data0_65536[(alu4+1)] = ((alu9*val15)+(alu10*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const E_2_32_8_16_2_4_dispatch_67 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+256)];
  var val1 = data1_196608[(alu0+257)];
  var val2 = data1_196608[(alu0+288)];
  var val3 = data1_196608[(alu0+289)];
  var val4 = data1_196608[(alu0+320)];
  var val5 = data1_196608[(alu0+321)];
  var val6 = data1_196608[(alu0+352)];
  var val7 = data1_196608[(alu0+353)];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val12);
  var alu10 = (val1*alu5*val13);
  var alu11 = (val2*alu6*val12);
  var alu12 = (val3*alu6*val13);
  var alu13 = (val4*alu7*val12);
  var alu14 = (val5*alu7*val13);
  var alu15 = (val6*alu8*val12);
  var alu16 = (val7*alu8*val13);
  data0_65536[alu4] = ((alu9*val14)-(alu10*val16));
  data0_65536[(alu4+1)] = ((alu10*val15)+(alu9*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const r_8_8_4_8_16_4_4_32_dispatch_68 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_524288:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
  var cast2 = bitcast<i32>((cast1<<13u));
  var cast3 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<10u))+bitcast<i32>((cast3<<7u))+cast2);
  var val0 = data1_65536[alu0];
  var val1 = data1_65536[(alu0+1)];
  var val2 = data1_65536[(alu0+10)];
  var val3 = data1_65536[(alu0+100)];
  var val4 = data1_65536[(alu0+101)];
  var val5 = data1_65536[(alu0+102)];
  var val6 = data1_65536[(alu0+103)];
  var val7 = data1_65536[(alu0+104)];
  var val8 = data1_65536[(alu0+105)];
  var val9 = data1_65536[(alu0+106)];
  var val10 = data1_65536[(alu0+107)];
  var val11 = data1_65536[(alu0+108)];
  var val12 = data1_65536[(alu0+109)];
  var val13 = data1_65536[(alu0+11)];
  var val14 = data1_65536[(alu0+110)];
  var val15 = data1_65536[(alu0+111)];
  var val16 = data1_65536[(alu0+112)];
  var val17 = data1_65536[(alu0+113)];
  var val18 = data1_65536[(alu0+114)];
  var val19 = data1_65536[(alu0+115)];
  var val20 = data1_65536[(alu0+116)];
  var val21 = data1_65536[(alu0+117)];
  var val22 = data1_65536[(alu0+118)];
  var val23 = data1_65536[(alu0+119)];
  var val24 = data1_65536[(alu0+12)];
  var val25 = data1_65536[(alu0+120)];
  var val26 = data1_65536[(alu0+121)];
  var val27 = data1_65536[(alu0+122)];
  var val28 = data1_65536[(alu0+123)];
  var val29 = data1_65536[(alu0+124)];
  var val30 = data1_65536[(alu0+125)];
  var val31 = data1_65536[(alu0+126)];
  var val32 = data1_65536[(alu0+127)];
  var val33 = data1_65536[(alu0+13)];
  var val34 = data1_65536[(alu0+14)];
  var val35 = data1_65536[(alu0+15)];
  var val36 = data1_65536[(alu0+16)];
  var val37 = data1_65536[(alu0+17)];
  var val38 = data1_65536[(alu0+18)];
  var val39 = data1_65536[(alu0+19)];
  var val40 = data1_65536[(alu0+2)];
  var val41 = data1_65536[(alu0+20)];
  var val42 = data1_65536[(alu0+21)];
  var val43 = data1_65536[(alu0+22)];
  var val44 = data1_65536[(alu0+23)];
  var val45 = data1_65536[(alu0+24)];
  var val46 = data1_65536[(alu0+25)];
  var val47 = data1_65536[(alu0+26)];
  var val48 = data1_65536[(alu0+27)];
  var val49 = data1_65536[(alu0+28)];
  var val50 = data1_65536[(alu0+29)];
  var val51 = data1_65536[(alu0+3)];
  var val52 = data1_65536[(alu0+30)];
  var val53 = data1_65536[(alu0+31)];
  var val54 = data1_65536[(alu0+32)];
  var val55 = data1_65536[(alu0+33)];
  var val56 = data1_65536[(alu0+34)];
  var val57 = data1_65536[(alu0+35)];
  var val58 = data1_65536[(alu0+36)];
  var val59 = data1_65536[(alu0+37)];
  var val60 = data1_65536[(alu0+38)];
  var val61 = data1_65536[(alu0+39)];
  var val62 = data1_65536[(alu0+4)];
  var val63 = data1_65536[(alu0+40)];
  var val64 = data1_65536[(alu0+41)];
  var val65 = data1_65536[(alu0+42)];
  var val66 = data1_65536[(alu0+43)];
  var val67 = data1_65536[(alu0+44)];
  var val68 = data1_65536[(alu0+45)];
  var val69 = data1_65536[(alu0+46)];
  var val70 = data1_65536[(alu0+47)];
  var val71 = data1_65536[(alu0+48)];
  var val72 = data1_65536[(alu0+49)];
  var val73 = data1_65536[(alu0+5)];
  var val74 = data1_65536[(alu0+50)];
  var val75 = data1_65536[(alu0+51)];
  var val76 = data1_65536[(alu0+52)];
  var val77 = data1_65536[(alu0+53)];
  var val78 = data1_65536[(alu0+54)];
  var val79 = data1_65536[(alu0+55)];
  var val80 = data1_65536[(alu0+56)];
  var val81 = data1_65536[(alu0+57)];
  var val82 = data1_65536[(alu0+58)];
  var val83 = data1_65536[(alu0+59)];
  var val84 = data1_65536[(alu0+6)];
  var val85 = data1_65536[(alu0+60)];
  var val86 = data1_65536[(alu0+61)];
  var val87 = data1_65536[(alu0+62)];
  var val88 = data1_65536[(alu0+63)];
  var val89 = data1_65536[(alu0+64)];
  var val90 = data1_65536[(alu0+65)];
  var val91 = data1_65536[(alu0+66)];
  var val92 = data1_65536[(alu0+67)];
  var val93 = data1_65536[(alu0+68)];
  var val94 = data1_65536[(alu0+69)];
  var val95 = data1_65536[(alu0+7)];
  var val96 = data1_65536[(alu0+70)];
  var val97 = data1_65536[(alu0+71)];
  var val98 = data1_65536[(alu0+72)];
  var val99 = data1_65536[(alu0+73)];
  var val100 = data1_65536[(alu0+74)];
  var val101 = data1_65536[(alu0+75)];
  var val102 = data1_65536[(alu0+76)];
  var val103 = data1_65536[(alu0+77)];
  var val104 = data1_65536[(alu0+78)];
  var val105 = data1_65536[(alu0+79)];
  var val106 = data1_65536[(alu0+8)];
  var val107 = data1_65536[(alu0+80)];
  var val108 = data1_65536[(alu0+81)];
  var val109 = data1_65536[(alu0+82)];
  var val110 = data1_65536[(alu0+83)];
  var val111 = data1_65536[(alu0+84)];
  var val112 = data1_65536[(alu0+85)];
  var val113 = data1_65536[(alu0+86)];
  var val114 = data1_65536[(alu0+87)];
  var val115 = data1_65536[(alu0+88)];
  var val116 = data1_65536[(alu0+89)];
  var val117 = data1_65536[(alu0+9)];
  var val118 = data1_65536[(alu0+90)];
  var val119 = data1_65536[(alu0+91)];
  var val120 = data1_65536[(alu0+92)];
  var val121 = data1_65536[(alu0+93)];
  var val122 = data1_65536[(alu0+94)];
  var val123 = data1_65536[(alu0+95)];
  var val124 = data1_65536[(alu0+96)];
  var val125 = data1_65536[(alu0+97)];
  var val126 = data1_65536[(alu0+98)];
  var val127 = data1_65536[(alu0+99)];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast4 = bitcast<u32>(gidx0);
  var cast5 = bitcast<u32>(lidx1);
  var alu1 = (bitcast<i32>((cast4<<11u))+bitcast<i32>((cast5<<7u))+cast2);
  var val128 = data2_65536[alu1];
  var val129 = data2_65536[(alu1+1)];
  var val130 = data2_65536[(alu1+10)];
  var val131 = data2_65536[(alu1+100)];
  var val132 = data2_65536[(alu1+101)];
  var val133 = data2_65536[(alu1+102)];
  var val134 = data2_65536[(alu1+103)];
  var val135 = data2_65536[(alu1+104)];
  var val136 = data2_65536[(alu1+105)];
  var val137 = data2_65536[(alu1+106)];
  var val138 = data2_65536[(alu1+107)];
  var val139 = data2_65536[(alu1+108)];
  var val140 = data2_65536[(alu1+109)];
  var val141 = data2_65536[(alu1+11)];
  var val142 = data2_65536[(alu1+110)];
  var val143 = data2_65536[(alu1+111)];
  var val144 = data2_65536[(alu1+112)];
  var val145 = data2_65536[(alu1+113)];
  var val146 = data2_65536[(alu1+114)];
  var val147 = data2_65536[(alu1+115)];
  var val148 = data2_65536[(alu1+116)];
  var val149 = data2_65536[(alu1+117)];
  var val150 = data2_65536[(alu1+118)];
  var val151 = data2_65536[(alu1+119)];
  var val152 = data2_65536[(alu1+12)];
  var val153 = data2_65536[(alu1+120)];
  var val154 = data2_65536[(alu1+121)];
  var val155 = data2_65536[(alu1+122)];
  var val156 = data2_65536[(alu1+123)];
  var val157 = data2_65536[(alu1+124)];
  var val158 = data2_65536[(alu1+125)];
  var val159 = data2_65536[(alu1+126)];
  var val160 = data2_65536[(alu1+127)];
  var val161 = data2_65536[(alu1+13)];
  var val162 = data2_65536[(alu1+14)];
  var val163 = data2_65536[(alu1+15)];
  var val164 = data2_65536[(alu1+16)];
  var val165 = data2_65536[(alu1+17)];
  var val166 = data2_65536[(alu1+18)];
  var val167 = data2_65536[(alu1+19)];
  var val168 = data2_65536[(alu1+2)];
  var val169 = data2_65536[(alu1+20)];
  var val170 = data2_65536[(alu1+21)];
  var val171 = data2_65536[(alu1+22)];
  var val172 = data2_65536[(alu1+23)];
  var val173 = data2_65536[(alu1+24)];
  var val174 = data2_65536[(alu1+25)];
  var val175 = data2_65536[(alu1+26)];
  var val176 = data2_65536[(alu1+27)];
  var val177 = data2_65536[(alu1+28)];
  var val178 = data2_65536[(alu1+29)];
  var val179 = data2_65536[(alu1+3)];
  var val180 = data2_65536[(alu1+30)];
  var val181 = data2_65536[(alu1+31)];
  var val182 = data2_65536[(alu1+32)];
  var val183 = data2_65536[(alu1+33)];
  var val184 = data2_65536[(alu1+34)];
  var val185 = data2_65536[(alu1+35)];
  var val186 = data2_65536[(alu1+36)];
  var val187 = data2_65536[(alu1+37)];
  var val188 = data2_65536[(alu1+38)];
  var val189 = data2_65536[(alu1+39)];
  var val190 = data2_65536[(alu1+4)];
  var val191 = data2_65536[(alu1+40)];
  var val192 = data2_65536[(alu1+41)];
  var val193 = data2_65536[(alu1+42)];
  var val194 = data2_65536[(alu1+43)];
  var val195 = data2_65536[(alu1+44)];
  var val196 = data2_65536[(alu1+45)];
  var val197 = data2_65536[(alu1+46)];
  var val198 = data2_65536[(alu1+47)];
  var val199 = data2_65536[(alu1+48)];
  var val200 = data2_65536[(alu1+49)];
  var val201 = data2_65536[(alu1+5)];
  var val202 = data2_65536[(alu1+50)];
  var val203 = data2_65536[(alu1+51)];
  var val204 = data2_65536[(alu1+52)];
  var val205 = data2_65536[(alu1+53)];
  var val206 = data2_65536[(alu1+54)];
  var val207 = data2_65536[(alu1+55)];
  var val208 = data2_65536[(alu1+56)];
  var val209 = data2_65536[(alu1+57)];
  var val210 = data2_65536[(alu1+58)];
  var val211 = data2_65536[(alu1+59)];
  var val212 = data2_65536[(alu1+6)];
  var val213 = data2_65536[(alu1+60)];
  var val214 = data2_65536[(alu1+61)];
  var val215 = data2_65536[(alu1+62)];
  var val216 = data2_65536[(alu1+63)];
  var val217 = data2_65536[(alu1+64)];
  var val218 = data2_65536[(alu1+65)];
  var val219 = data2_65536[(alu1+66)];
  var val220 = data2_65536[(alu1+67)];
  var val221 = data2_65536[(alu1+68)];
  var val222 = data2_65536[(alu1+69)];
  var val223 = data2_65536[(alu1+7)];
  var val224 = data2_65536[(alu1+70)];
  var val225 = data2_65536[(alu1+71)];
  var val226 = data2_65536[(alu1+72)];
  var val227 = data2_65536[(alu1+73)];
  var val228 = data2_65536[(alu1+74)];
  var val229 = data2_65536[(alu1+75)];
  var val230 = data2_65536[(alu1+76)];
  var val231 = data2_65536[(alu1+77)];
  var val232 = data2_65536[(alu1+78)];
  var val233 = data2_65536[(alu1+79)];
  var val234 = data2_65536[(alu1+8)];
  var val235 = data2_65536[(alu1+80)];
  var val236 = data2_65536[(alu1+81)];
  var val237 = data2_65536[(alu1+82)];
  var val238 = data2_65536[(alu1+83)];
  var val239 = data2_65536[(alu1+84)];
  var val240 = data2_65536[(alu1+85)];
  var val241 = data2_65536[(alu1+86)];
  var val242 = data2_65536[(alu1+87)];
  var val243 = data2_65536[(alu1+88)];
  var val244 = data2_65536[(alu1+89)];
  var val245 = data2_65536[(alu1+9)];
  var val246 = data2_65536[(alu1+90)];
  var val247 = data2_65536[(alu1+91)];
  var val248 = data2_65536[(alu1+92)];
  var val249 = data2_65536[(alu1+93)];
  var val250 = data2_65536[(alu1+94)];
  var val251 = data2_65536[(alu1+95)];
  var val252 = data2_65536[(alu1+96)];
  var val253 = data2_65536[(alu1+97)];
  var val254 = data2_65536[(alu1+98)];
  var val255 = data2_65536[(alu1+99)];
  var alu2 = (bitcast<i32>((cast4<<6u))+bitcast<i32>((cast5<<2u))+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast1<<16u)));
  data0_524288[alu2] = (((val0*val128)+(val1*val129)+(val40*val168)+(val51*val179)+(val62*val190)+(val73*val201)+(val84*val212)+(val95*val223)+(val106*val234)+(val117*val245)+(val2*val130)+(val13*val141)+(val24*val152)+(val33*val161)+(val34*val162)+(val35*val163)+(val36*val164)+(val37*val165)+(val38*val166)+(val39*val167)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val173)+(val46*val174)+(val47*val175)+(val48*val176)+(val49*val177)+(val50*val178)+(val52*val180)+(val53*val181))*0.1767766922712326f);
  data0_524288[(alu2+1)] = (((val0*val182)+(val1*val183)+(val40*val184)+(val51*val185)+(val62*val186)+(val73*val187)+(val84*val188)+(val95*val189)+(val106*val191)+(val117*val192)+(val2*val193)+(val13*val194)+(val24*val195)+(val33*val196)+(val34*val197)+(val35*val198)+(val36*val199)+(val37*val200)+(val38*val202)+(val39*val203)+(val41*val204)+(val42*val205)+(val43*val206)+(val44*val207)+(val45*val208)+(val46*val209)+(val47*val210)+(val48*val211)+(val49*val213)+(val50*val214)+(val52*val215)+(val53*val216))*0.1767766922712326f);
  data0_524288[(alu2+2)] = (((val0*val217)+(val1*val218)+(val40*val219)+(val51*val220)+(val62*val221)+(val73*val222)+(val84*val224)+(val95*val225)+(val106*val226)+(val117*val227)+(val2*val228)+(val13*val229)+(val24*val230)+(val33*val231)+(val34*val232)+(val35*val233)+(val36*val235)+(val37*val236)+(val38*val237)+(val39*val238)+(val41*val239)+(val42*val240)+(val43*val241)+(val44*val242)+(val45*val243)+(val46*val244)+(val47*val246)+(val48*val247)+(val49*val248)+(val50*val249)+(val52*val250)+(val53*val251))*0.1767766922712326f);
  data0_524288[(alu2+256)] = (((val54*val128)+(val55*val129)+(val56*val168)+(val57*val179)+(val58*val190)+(val59*val201)+(val60*val212)+(val61*val223)+(val63*val234)+(val64*val245)+(val65*val130)+(val66*val141)+(val67*val152)+(val68*val161)+(val69*val162)+(val70*val163)+(val71*val164)+(val72*val165)+(val74*val166)+(val75*val167)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val173)+(val81*val174)+(val82*val175)+(val83*val176)+(val85*val177)+(val86*val178)+(val87*val180)+(val88*val181))*0.1767766922712326f);
  data0_524288[(alu2+257)] = (((val54*val182)+(val55*val183)+(val56*val184)+(val57*val185)+(val58*val186)+(val59*val187)+(val60*val188)+(val61*val189)+(val63*val191)+(val64*val192)+(val65*val193)+(val66*val194)+(val67*val195)+(val68*val196)+(val69*val197)+(val70*val198)+(val71*val199)+(val72*val200)+(val74*val202)+(val75*val203)+(val76*val204)+(val77*val205)+(val78*val206)+(val79*val207)+(val80*val208)+(val81*val209)+(val82*val210)+(val83*val211)+(val85*val213)+(val86*val214)+(val87*val215)+(val88*val216))*0.1767766922712326f);
  data0_524288[(alu2+258)] = (((val54*val217)+(val55*val218)+(val56*val219)+(val57*val220)+(val58*val221)+(val59*val222)+(val60*val224)+(val61*val225)+(val63*val226)+(val64*val227)+(val65*val228)+(val66*val229)+(val67*val230)+(val68*val231)+(val69*val232)+(val70*val233)+(val71*val235)+(val72*val236)+(val74*val237)+(val75*val238)+(val76*val239)+(val77*val240)+(val78*val241)+(val79*val242)+(val80*val243)+(val81*val244)+(val82*val246)+(val83*val247)+(val85*val248)+(val86*val249)+(val87*val250)+(val88*val251))*0.1767766922712326f);
  data0_524288[(alu2+259)] = (((val54*val252)+(val55*val253)+(val56*val254)+(val57*val255)+(val58*val131)+(val59*val132)+(val60*val133)+(val61*val134)+(val63*val135)+(val64*val136)+(val65*val137)+(val66*val138)+(val67*val139)+(val68*val140)+(val69*val142)+(val70*val143)+(val71*val144)+(val72*val145)+(val74*val146)+(val75*val147)+(val76*val148)+(val77*val149)+(val78*val150)+(val79*val151)+(val80*val153)+(val81*val154)+(val82*val155)+(val83*val156)+(val85*val157)+(val86*val158)+(val87*val159)+(val88*val160))*0.1767766922712326f);
  data0_524288[(alu2+3)] = (((val0*val252)+(val1*val253)+(val40*val254)+(val51*val255)+(val62*val131)+(val73*val132)+(val84*val133)+(val95*val134)+(val106*val135)+(val117*val136)+(val2*val137)+(val13*val138)+(val24*val139)+(val33*val140)+(val34*val142)+(val35*val143)+(val36*val144)+(val37*val145)+(val38*val146)+(val39*val147)+(val41*val148)+(val42*val149)+(val43*val150)+(val44*val151)+(val45*val153)+(val46*val154)+(val47*val155)+(val48*val156)+(val49*val157)+(val50*val158)+(val52*val159)+(val53*val160))*0.1767766922712326f);
  data0_524288[(alu2+512)] = (((val89*val128)+(val90*val129)+(val91*val168)+(val92*val179)+(val93*val190)+(val94*val201)+(val96*val212)+(val97*val223)+(val98*val234)+(val99*val245)+(val100*val130)+(val101*val141)+(val102*val152)+(val103*val161)+(val104*val162)+(val105*val163)+(val107*val164)+(val108*val165)+(val109*val166)+(val110*val167)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val173)+(val116*val174)+(val118*val175)+(val119*val176)+(val120*val177)+(val121*val178)+(val122*val180)+(val123*val181))*0.1767766922712326f);
  data0_524288[(alu2+513)] = (((val89*val182)+(val90*val183)+(val91*val184)+(val92*val185)+(val93*val186)+(val94*val187)+(val96*val188)+(val97*val189)+(val98*val191)+(val99*val192)+(val100*val193)+(val101*val194)+(val102*val195)+(val103*val196)+(val104*val197)+(val105*val198)+(val107*val199)+(val108*val200)+(val109*val202)+(val110*val203)+(val111*val204)+(val112*val205)+(val113*val206)+(val114*val207)+(val115*val208)+(val116*val209)+(val118*val210)+(val119*val211)+(val120*val213)+(val121*val214)+(val122*val215)+(val123*val216))*0.1767766922712326f);
  data0_524288[(alu2+514)] = (((val89*val217)+(val90*val218)+(val91*val219)+(val92*val220)+(val93*val221)+(val94*val222)+(val96*val224)+(val97*val225)+(val98*val226)+(val99*val227)+(val100*val228)+(val101*val229)+(val102*val230)+(val103*val231)+(val104*val232)+(val105*val233)+(val107*val235)+(val108*val236)+(val109*val237)+(val110*val238)+(val111*val239)+(val112*val240)+(val113*val241)+(val114*val242)+(val115*val243)+(val116*val244)+(val118*val246)+(val119*val247)+(val120*val248)+(val121*val249)+(val122*val250)+(val123*val251))*0.1767766922712326f);
  data0_524288[(alu2+515)] = (((val89*val252)+(val90*val253)+(val91*val254)+(val92*val255)+(val93*val131)+(val94*val132)+(val96*val133)+(val97*val134)+(val98*val135)+(val99*val136)+(val100*val137)+(val101*val138)+(val102*val139)+(val103*val140)+(val104*val142)+(val105*val143)+(val107*val144)+(val108*val145)+(val109*val146)+(val110*val147)+(val111*val148)+(val112*val149)+(val113*val150)+(val114*val151)+(val115*val153)+(val116*val154)+(val118*val155)+(val119*val156)+(val120*val157)+(val121*val158)+(val122*val159)+(val123*val160))*0.1767766922712326f);
  data0_524288[(alu2+768)] = (((val124*val128)+(val125*val129)+(val126*val168)+(val127*val179)+(val3*val190)+(val4*val201)+(val5*val212)+(val6*val223)+(val7*val234)+(val8*val245)+(val9*val130)+(val10*val141)+(val11*val152)+(val12*val161)+(val14*val162)+(val15*val163)+(val16*val164)+(val17*val165)+(val18*val166)+(val19*val167)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val173)+(val26*val174)+(val27*val175)+(val28*val176)+(val29*val177)+(val30*val178)+(val31*val180)+(val32*val181))*0.1767766922712326f);
  data0_524288[(alu2+769)] = (((val124*val182)+(val125*val183)+(val126*val184)+(val127*val185)+(val3*val186)+(val4*val187)+(val5*val188)+(val6*val189)+(val7*val191)+(val8*val192)+(val9*val193)+(val10*val194)+(val11*val195)+(val12*val196)+(val14*val197)+(val15*val198)+(val16*val199)+(val17*val200)+(val18*val202)+(val19*val203)+(val20*val204)+(val21*val205)+(val22*val206)+(val23*val207)+(val25*val208)+(val26*val209)+(val27*val210)+(val28*val211)+(val29*val213)+(val30*val214)+(val31*val215)+(val32*val216))*0.1767766922712326f);
  data0_524288[(alu2+770)] = (((val124*val217)+(val125*val218)+(val126*val219)+(val127*val220)+(val3*val221)+(val4*val222)+(val5*val224)+(val6*val225)+(val7*val226)+(val8*val227)+(val9*val228)+(val10*val229)+(val11*val230)+(val12*val231)+(val14*val232)+(val15*val233)+(val16*val235)+(val17*val236)+(val18*val237)+(val19*val238)+(val20*val239)+(val21*val240)+(val22*val241)+(val23*val242)+(val25*val243)+(val26*val244)+(val27*val246)+(val28*val247)+(val29*val248)+(val30*val249)+(val31*val250)+(val32*val251))*0.1767766922712326f);
  data0_524288[(alu2+771)] = (((val124*val252)+(val125*val253)+(val126*val254)+(val127*val255)+(val3*val131)+(val4*val132)+(val5*val133)+(val6*val134)+(val7*val135)+(val8*val136)+(val9*val137)+(val10*val138)+(val11*val139)+(val12*val140)+(val14*val142)+(val15*val143)+(val16*val144)+(val17*val145)+(val18*val146)+(val19*val147)+(val20*val148)+(val21*val149)+(val22*val150)+(val23*val151)+(val25*val153)+(val26*val154)+(val27*val155)+(val28*val156)+(val29*val157)+(val30*val158)+(val31*val159)+(val32*val160))*0.1767766922712326f);
}`;

const r_2048_16_16_dispatch_69 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = (f32(-INFINITY));
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val0 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    var val1 = buf0[0];
    var alu1 = select(val1,val0,(val1<val0));
    buf0[0] = alu1;
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = (f32(-INFINITY));
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf1[0];
    var val3 = buf2[Lidx102];
    var alu7 = select(val2,val3,(val2<val3));
    buf1[0] = alu7;
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_2048_16_16_dispatch_70 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var val0 = data2_2048[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val1 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    buf0[0] = (buf0[0]+exp2(((val1-val0)*1.4426950216293335f)));
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf2[Lidx102];
    buf1[0] = (buf1[0]+val2);
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_8_4_16_8_4_4_64_4_dispatch_71 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2048:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_196608:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u))+bitcast<i32>((cast1<<8u)));
  var val0 = data2_2048[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_2048[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_2048[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_2048[alu3];
  var lidx1 = i32(lindex.y); /* 8 */
  var cast3 = bitcast<i32>((bitcast<u32>(lidx1)<<2u));
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var alu20 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast2<<10u))+bitcast<i32>((bitcast<u32>(Lidx0)<<2u))+bitcast<i32>((cast1<<16u)));
    var val4 = data1_524288[alu20];
    var val5 = data1_524288[(alu20+1)];
    var val6 = data1_524288[(alu20+2)];
    var val7 = data1_524288[(alu20+256)];
    var val8 = data1_524288[(alu20+257)];
    var val9 = data1_524288[(alu20+258)];
    var val10 = data1_524288[(alu20+259)];
    var val11 = data1_524288[(alu20+3)];
    var val12 = data1_524288[(alu20+512)];
    var val13 = data1_524288[(alu20+513)];
    var val14 = data1_524288[(alu20+514)];
    var val15 = data1_524288[(alu20+515)];
    var val16 = data1_524288[(alu20+768)];
    var val17 = data1_524288[(alu20+769)];
    var val18 = data1_524288[(alu20+770)];
    var val19 = data1_524288[(alu20+771)];
    var alu21 = (bitcast<i32>((cast1<<5u))+cast3+(Lidx0*3072));
    var val20 = data4_196608[(alu21+1280)];
    var val21 = data4_196608[(alu21+1281)];
    var val22 = data4_196608[(alu21+1282)];
    var val23 = data4_196608[(alu21+1283)];
    var val24 = data4_196608[(alu21+2048)];
    var val25 = data4_196608[(alu21+2049)];
    var val26 = data4_196608[(alu21+2050)];
    var val27 = data4_196608[(alu21+2051)];
    var val28 = data4_196608[(alu21+2816)];
    var val29 = data4_196608[(alu21+2817)];
    var val30 = data4_196608[(alu21+2818)];
    var val31 = data4_196608[(alu21+2819)];
    var val32 = data4_196608[(alu21+512)];
    var val33 = data4_196608[(alu21+513)];
    var val34 = data4_196608[(alu21+514)];
    var val35 = data4_196608[(alu21+515)];
    var alu22 = exp2(((val4-val0)*1.4426950216293335f));
    var alu23 = exp2(((val5-val0)*1.4426950216293335f));
    var alu24 = exp2(((val6-val0)*1.4426950216293335f));
    var alu25 = exp2(((val7-val1)*1.4426950216293335f));
    var alu26 = exp2(((val8-val1)*1.4426950216293335f));
    var alu27 = exp2(((val9-val1)*1.4426950216293335f));
    var alu28 = exp2(((val10-val1)*1.4426950216293335f));
    var alu29 = exp2(((val11-val0)*1.4426950216293335f));
    var alu30 = exp2(((val12-val2)*1.4426950216293335f));
    var alu31 = exp2(((val13-val2)*1.4426950216293335f));
    var alu32 = exp2(((val14-val2)*1.4426950216293335f));
    var alu33 = exp2(((val15-val2)*1.4426950216293335f));
    var alu34 = exp2(((val16-val3)*1.4426950216293335f));
    var alu35 = exp2(((val17-val3)*1.4426950216293335f));
    var alu36 = exp2(((val18-val3)*1.4426950216293335f));
    var alu37 = exp2(((val19-val3)*1.4426950216293335f));
    buf0[0] = (buf0[0]+(alu22*val32)+(alu23*val20)+(alu24*val24)+(alu29*val28));
    buf0[1] = (buf0[1]+(alu22*val33)+(alu23*val21)+(alu24*val25)+(alu29*val29));
    buf0[10] = (buf0[10]+(alu30*val34)+(alu31*val22)+(alu32*val26)+(alu33*val30));
    buf0[11] = (buf0[11]+(alu30*val35)+(alu31*val23)+(alu32*val27)+(alu33*val31));
    buf0[12] = (buf0[12]+(alu34*val32)+(alu35*val20)+(alu36*val24)+(alu37*val28));
    buf0[13] = (buf0[13]+(alu34*val33)+(alu35*val21)+(alu36*val25)+(alu37*val29));
    buf0[14] = (buf0[14]+(alu34*val34)+(alu35*val22)+(alu36*val26)+(alu37*val30));
    buf0[15] = (buf0[15]+(alu34*val35)+(alu35*val23)+(alu36*val27)+(alu37*val31));
    buf0[2] = (buf0[2]+(alu22*val34)+(alu23*val22)+(alu24*val26)+(alu29*val30));
    buf0[3] = (buf0[3]+(alu22*val35)+(alu23*val23)+(alu24*val27)+(alu29*val31));
    buf0[4] = (buf0[4]+(alu25*val32)+(alu26*val20)+(alu27*val24)+(alu28*val28));
    buf0[5] = (buf0[5]+(alu25*val33)+(alu26*val21)+(alu27*val25)+(alu28*val29));
    buf0[6] = (buf0[6]+(alu25*val34)+(alu26*val22)+(alu27*val26)+(alu28*val30));
    buf0[7] = (buf0[7]+(alu25*val35)+(alu26*val23)+(alu27*val27)+(alu28*val31));
    buf0[8] = (buf0[8]+(alu30*val32)+(alu31*val20)+(alu32*val24)+(alu33*val28));
    buf0[9] = (buf0[9]+(alu30*val33)+(alu31*val21)+(alu32*val25)+(alu33*val29));
  }
  var val36 = data3_2048[alu0];
  var val37 = data3_2048[alu1];
  var val38 = data3_2048[alu2];
  var val39 = data3_2048[alu3];
  var alu55 = (bitcast<i32>((cast0<<11u))+bitcast<i32>((cast2<<7u))+cast3+bitcast<i32>((cast1<<13u)));
  var alu56 = (1/val36);
  var alu57 = (1/val37);
  var alu58 = (1/val38);
  var alu59 = (1/val39);
  data0_65536[alu55] = (buf0[0]*alu56);
  data0_65536[(alu55+1)] = (buf0[1]*alu56);
  data0_65536[(alu55+2)] = (buf0[2]*alu56);
  data0_65536[(alu55+3)] = (buf0[3]*alu56);
  data0_65536[(alu55+32)] = (buf0[4]*alu57);
  data0_65536[(alu55+33)] = (buf0[5]*alu57);
  data0_65536[(alu55+34)] = (buf0[6]*alu57);
  data0_65536[(alu55+35)] = (buf0[7]*alu57);
  data0_65536[(alu55+64)] = (buf0[8]*alu58);
  data0_65536[(alu55+65)] = (buf0[9]*alu58);
  data0_65536[(alu55+66)] = (buf0[10]*alu58);
  data0_65536[(alu55+67)] = (buf0[11]*alu58);
  data0_65536[(alu55+96)] = (buf0[12]*alu59);
  data0_65536[(alu55+97)] = (buf0[13]*alu59);
  data0_65536[(alu55+98)] = (buf0[14]*alu59);
  data0_65536[(alu55+99)] = (buf0[15]*alu59);
}`;

const r_8_4_8_16_4_4_8_32_dispatch_72 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_65536:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0_0 = 0; Lidx0_0 < 8; Lidx0_0++) {
    var cast4 = bitcast<u32>(Lidx0_0);
    var alu16 = (bitcast<i32>((cast1<<10u))+bitcast<i32>((cast2<<7u))+bitcast<i32>((cast4<<13u)));
    var val0 = data3_65536[alu16];
    var val1 = data3_65536[(alu16+1)];
    var val2 = data3_65536[(alu16+10)];
    var val3 = data3_65536[(alu16+100)];
    var val4 = data3_65536[(alu16+101)];
    var val5 = data3_65536[(alu16+102)];
    var val6 = data3_65536[(alu16+103)];
    var val7 = data3_65536[(alu16+104)];
    var val8 = data3_65536[(alu16+105)];
    var val9 = data3_65536[(alu16+106)];
    var val10 = data3_65536[(alu16+107)];
    var val11 = data3_65536[(alu16+108)];
    var val12 = data3_65536[(alu16+109)];
    var val13 = data3_65536[(alu16+11)];
    var val14 = data3_65536[(alu16+110)];
    var val15 = data3_65536[(alu16+111)];
    var val16 = data3_65536[(alu16+112)];
    var val17 = data3_65536[(alu16+113)];
    var val18 = data3_65536[(alu16+114)];
    var val19 = data3_65536[(alu16+115)];
    var val20 = data3_65536[(alu16+116)];
    var val21 = data3_65536[(alu16+117)];
    var val22 = data3_65536[(alu16+118)];
    var val23 = data3_65536[(alu16+119)];
    var val24 = data3_65536[(alu16+12)];
    var val25 = data3_65536[(alu16+120)];
    var val26 = data3_65536[(alu16+121)];
    var val27 = data3_65536[(alu16+122)];
    var val28 = data3_65536[(alu16+123)];
    var val29 = data3_65536[(alu16+124)];
    var val30 = data3_65536[(alu16+125)];
    var val31 = data3_65536[(alu16+126)];
    var val32 = data3_65536[(alu16+127)];
    var val33 = data3_65536[(alu16+13)];
    var val34 = data3_65536[(alu16+14)];
    var val35 = data3_65536[(alu16+15)];
    var val36 = data3_65536[(alu16+16)];
    var val37 = data3_65536[(alu16+17)];
    var val38 = data3_65536[(alu16+18)];
    var val39 = data3_65536[(alu16+19)];
    var val40 = data3_65536[(alu16+2)];
    var val41 = data3_65536[(alu16+20)];
    var val42 = data3_65536[(alu16+21)];
    var val43 = data3_65536[(alu16+22)];
    var val44 = data3_65536[(alu16+23)];
    var val45 = data3_65536[(alu16+24)];
    var val46 = data3_65536[(alu16+25)];
    var val47 = data3_65536[(alu16+26)];
    var val48 = data3_65536[(alu16+27)];
    var val49 = data3_65536[(alu16+28)];
    var val50 = data3_65536[(alu16+29)];
    var val51 = data3_65536[(alu16+3)];
    var val52 = data3_65536[(alu16+30)];
    var val53 = data3_65536[(alu16+31)];
    var val54 = data3_65536[(alu16+32)];
    var val55 = data3_65536[(alu16+33)];
    var val56 = data3_65536[(alu16+34)];
    var val57 = data3_65536[(alu16+35)];
    var val58 = data3_65536[(alu16+36)];
    var val59 = data3_65536[(alu16+37)];
    var val60 = data3_65536[(alu16+38)];
    var val61 = data3_65536[(alu16+39)];
    var val62 = data3_65536[(alu16+4)];
    var val63 = data3_65536[(alu16+40)];
    var val64 = data3_65536[(alu16+41)];
    var val65 = data3_65536[(alu16+42)];
    var val66 = data3_65536[(alu16+43)];
    var val67 = data3_65536[(alu16+44)];
    var val68 = data3_65536[(alu16+45)];
    var val69 = data3_65536[(alu16+46)];
    var val70 = data3_65536[(alu16+47)];
    var val71 = data3_65536[(alu16+48)];
    var val72 = data3_65536[(alu16+49)];
    var val73 = data3_65536[(alu16+5)];
    var val74 = data3_65536[(alu16+50)];
    var val75 = data3_65536[(alu16+51)];
    var val76 = data3_65536[(alu16+52)];
    var val77 = data3_65536[(alu16+53)];
    var val78 = data3_65536[(alu16+54)];
    var val79 = data3_65536[(alu16+55)];
    var val80 = data3_65536[(alu16+56)];
    var val81 = data3_65536[(alu16+57)];
    var val82 = data3_65536[(alu16+58)];
    var val83 = data3_65536[(alu16+59)];
    var val84 = data3_65536[(alu16+6)];
    var val85 = data3_65536[(alu16+60)];
    var val86 = data3_65536[(alu16+61)];
    var val87 = data3_65536[(alu16+62)];
    var val88 = data3_65536[(alu16+63)];
    var val89 = data3_65536[(alu16+64)];
    var val90 = data3_65536[(alu16+65)];
    var val91 = data3_65536[(alu16+66)];
    var val92 = data3_65536[(alu16+67)];
    var val93 = data3_65536[(alu16+68)];
    var val94 = data3_65536[(alu16+69)];
    var val95 = data3_65536[(alu16+7)];
    var val96 = data3_65536[(alu16+70)];
    var val97 = data3_65536[(alu16+71)];
    var val98 = data3_65536[(alu16+72)];
    var val99 = data3_65536[(alu16+73)];
    var val100 = data3_65536[(alu16+74)];
    var val101 = data3_65536[(alu16+75)];
    var val102 = data3_65536[(alu16+76)];
    var val103 = data3_65536[(alu16+77)];
    var val104 = data3_65536[(alu16+78)];
    var val105 = data3_65536[(alu16+79)];
    var val106 = data3_65536[(alu16+8)];
    var val107 = data3_65536[(alu16+80)];
    var val108 = data3_65536[(alu16+81)];
    var val109 = data3_65536[(alu16+82)];
    var val110 = data3_65536[(alu16+83)];
    var val111 = data3_65536[(alu16+84)];
    var val112 = data3_65536[(alu16+85)];
    var val113 = data3_65536[(alu16+86)];
    var val114 = data3_65536[(alu16+87)];
    var val115 = data3_65536[(alu16+88)];
    var val116 = data3_65536[(alu16+89)];
    var val117 = data3_65536[(alu16+9)];
    var val118 = data3_65536[(alu16+90)];
    var val119 = data3_65536[(alu16+91)];
    var val120 = data3_65536[(alu16+92)];
    var val121 = data3_65536[(alu16+93)];
    var val122 = data3_65536[(alu16+94)];
    var val123 = data3_65536[(alu16+95)];
    var val124 = data3_65536[(alu16+96)];
    var val125 = data3_65536[(alu16+97)];
    var val126 = data3_65536[(alu16+98)];
    var val127 = data3_65536[(alu16+99)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast4<<5u)));
    var val128 = data4_65536[alu17];
    var val129 = data4_65536[(alu17+1)];
    var val130 = data4_65536[(alu17+10)];
    var val131 = data4_65536[(alu17+11)];
    var val132 = data4_65536[(alu17+12)];
    var val133 = data4_65536[(alu17+13)];
    var val134 = data4_65536[(alu17+14)];
    var val135 = data4_65536[(alu17+15)];
    var val136 = data4_65536[(alu17+16)];
    var val137 = data4_65536[(alu17+17)];
    var val138 = data4_65536[(alu17+18)];
    var val139 = data4_65536[(alu17+19)];
    var val140 = data4_65536[(alu17+2)];
    var val141 = data4_65536[(alu17+20)];
    var val142 = data4_65536[(alu17+21)];
    var val143 = data4_65536[(alu17+22)];
    var val144 = data4_65536[(alu17+23)];
    var val145 = data4_65536[(alu17+24)];
    var val146 = data4_65536[(alu17+25)];
    var val147 = data4_65536[(alu17+256)];
    var val148 = data4_65536[(alu17+257)];
    var val149 = data4_65536[(alu17+258)];
    var val150 = data4_65536[(alu17+259)];
    var val151 = data4_65536[(alu17+26)];
    var val152 = data4_65536[(alu17+260)];
    var val153 = data4_65536[(alu17+261)];
    var val154 = data4_65536[(alu17+262)];
    var val155 = data4_65536[(alu17+263)];
    var val156 = data4_65536[(alu17+264)];
    var val157 = data4_65536[(alu17+265)];
    var val158 = data4_65536[(alu17+266)];
    var val159 = data4_65536[(alu17+267)];
    var val160 = data4_65536[(alu17+268)];
    var val161 = data4_65536[(alu17+269)];
    var val162 = data4_65536[(alu17+27)];
    var val163 = data4_65536[(alu17+270)];
    var val164 = data4_65536[(alu17+271)];
    var val165 = data4_65536[(alu17+272)];
    var val166 = data4_65536[(alu17+273)];
    var val167 = data4_65536[(alu17+274)];
    var val168 = data4_65536[(alu17+275)];
    var val169 = data4_65536[(alu17+276)];
    var val170 = data4_65536[(alu17+277)];
    var val171 = data4_65536[(alu17+278)];
    var val172 = data4_65536[(alu17+279)];
    var val173 = data4_65536[(alu17+28)];
    var val174 = data4_65536[(alu17+280)];
    var val175 = data4_65536[(alu17+281)];
    var val176 = data4_65536[(alu17+282)];
    var val177 = data4_65536[(alu17+283)];
    var val178 = data4_65536[(alu17+284)];
    var val179 = data4_65536[(alu17+285)];
    var val180 = data4_65536[(alu17+286)];
    var val181 = data4_65536[(alu17+287)];
    var val182 = data4_65536[(alu17+29)];
    var val183 = data4_65536[(alu17+3)];
    var val184 = data4_65536[(alu17+30)];
    var val185 = data4_65536[(alu17+31)];
    var val186 = data4_65536[(alu17+4)];
    var val187 = data4_65536[(alu17+5)];
    var val188 = data4_65536[(alu17+512)];
    var val189 = data4_65536[(alu17+513)];
    var val190 = data4_65536[(alu17+514)];
    var val191 = data4_65536[(alu17+515)];
    var val192 = data4_65536[(alu17+516)];
    var val193 = data4_65536[(alu17+517)];
    var val194 = data4_65536[(alu17+518)];
    var val195 = data4_65536[(alu17+519)];
    var val196 = data4_65536[(alu17+520)];
    var val197 = data4_65536[(alu17+521)];
    var val198 = data4_65536[(alu17+522)];
    var val199 = data4_65536[(alu17+523)];
    var val200 = data4_65536[(alu17+524)];
    var val201 = data4_65536[(alu17+525)];
    var val202 = data4_65536[(alu17+526)];
    var val203 = data4_65536[(alu17+527)];
    var val204 = data4_65536[(alu17+528)];
    var val205 = data4_65536[(alu17+529)];
    var val206 = data4_65536[(alu17+530)];
    var val207 = data4_65536[(alu17+531)];
    var val208 = data4_65536[(alu17+532)];
    var val209 = data4_65536[(alu17+533)];
    var val210 = data4_65536[(alu17+534)];
    var val211 = data4_65536[(alu17+535)];
    var val212 = data4_65536[(alu17+536)];
    var val213 = data4_65536[(alu17+537)];
    var val214 = data4_65536[(alu17+538)];
    var val215 = data4_65536[(alu17+539)];
    var val216 = data4_65536[(alu17+540)];
    var val217 = data4_65536[(alu17+541)];
    var val218 = data4_65536[(alu17+542)];
    var val219 = data4_65536[(alu17+543)];
    var val220 = data4_65536[(alu17+6)];
    var val221 = data4_65536[(alu17+7)];
    var val222 = data4_65536[(alu17+768)];
    var val223 = data4_65536[(alu17+769)];
    var val224 = data4_65536[(alu17+770)];
    var val225 = data4_65536[(alu17+771)];
    var val226 = data4_65536[(alu17+772)];
    var val227 = data4_65536[(alu17+773)];
    var val228 = data4_65536[(alu17+774)];
    var val229 = data4_65536[(alu17+775)];
    var val230 = data4_65536[(alu17+776)];
    var val231 = data4_65536[(alu17+777)];
    var val232 = data4_65536[(alu17+778)];
    var val233 = data4_65536[(alu17+779)];
    var val234 = data4_65536[(alu17+780)];
    var val235 = data4_65536[(alu17+781)];
    var val236 = data4_65536[(alu17+782)];
    var val237 = data4_65536[(alu17+783)];
    var val238 = data4_65536[(alu17+784)];
    var val239 = data4_65536[(alu17+785)];
    var val240 = data4_65536[(alu17+786)];
    var val241 = data4_65536[(alu17+787)];
    var val242 = data4_65536[(alu17+788)];
    var val243 = data4_65536[(alu17+789)];
    var val244 = data4_65536[(alu17+790)];
    var val245 = data4_65536[(alu17+791)];
    var val246 = data4_65536[(alu17+792)];
    var val247 = data4_65536[(alu17+793)];
    var val248 = data4_65536[(alu17+794)];
    var val249 = data4_65536[(alu17+795)];
    var val250 = data4_65536[(alu17+796)];
    var val251 = data4_65536[(alu17+797)];
    var val252 = data4_65536[(alu17+798)];
    var val253 = data4_65536[(alu17+799)];
    var val254 = data4_65536[(alu17+8)];
    var val255 = data4_65536[(alu17+9)];
    buf0[0] = (buf0[0]+(val0*val128)+(val1*val129)+(val40*val140)+(val51*val183)+(val62*val186)+(val73*val187)+(val84*val220)+(val95*val221)+(val106*val254)+(val117*val255)+(val2*val130)+(val13*val131)+(val24*val132)+(val33*val133)+(val34*val134)+(val35*val135)+(val36*val136)+(val37*val137)+(val38*val138)+(val39*val139)+(val41*val141)+(val42*val142)+(val43*val143)+(val44*val144)+(val45*val145)+(val46*val146)+(val47*val151)+(val48*val162)+(val49*val173)+(val50*val182)+(val52*val184)+(val53*val185));
    buf0[1] = (buf0[1]+(val0*val147)+(val1*val148)+(val40*val149)+(val51*val150)+(val62*val152)+(val73*val153)+(val84*val154)+(val95*val155)+(val106*val156)+(val117*val157)+(val2*val158)+(val13*val159)+(val24*val160)+(val33*val161)+(val34*val163)+(val35*val164)+(val36*val165)+(val37*val166)+(val38*val167)+(val39*val168)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val174)+(val46*val175)+(val47*val176)+(val48*val177)+(val49*val178)+(val50*val179)+(val52*val180)+(val53*val181));
    buf0[10] = (buf0[10]+(val89*val188)+(val90*val189)+(val91*val190)+(val92*val191)+(val93*val192)+(val94*val193)+(val96*val194)+(val97*val195)+(val98*val196)+(val99*val197)+(val100*val198)+(val101*val199)+(val102*val200)+(val103*val201)+(val104*val202)+(val105*val203)+(val107*val204)+(val108*val205)+(val109*val206)+(val110*val207)+(val111*val208)+(val112*val209)+(val113*val210)+(val114*val211)+(val115*val212)+(val116*val213)+(val118*val214)+(val119*val215)+(val120*val216)+(val121*val217)+(val122*val218)+(val123*val219));
    buf0[11] = (buf0[11]+(val89*val222)+(val90*val223)+(val91*val224)+(val92*val225)+(val93*val226)+(val94*val227)+(val96*val228)+(val97*val229)+(val98*val230)+(val99*val231)+(val100*val232)+(val101*val233)+(val102*val234)+(val103*val235)+(val104*val236)+(val105*val237)+(val107*val238)+(val108*val239)+(val109*val240)+(val110*val241)+(val111*val242)+(val112*val243)+(val113*val244)+(val114*val245)+(val115*val246)+(val116*val247)+(val118*val248)+(val119*val249)+(val120*val250)+(val121*val251)+(val122*val252)+(val123*val253));
    buf0[12] = (buf0[12]+(val124*val128)+(val125*val129)+(val126*val140)+(val127*val183)+(val3*val186)+(val4*val187)+(val5*val220)+(val6*val221)+(val7*val254)+(val8*val255)+(val9*val130)+(val10*val131)+(val11*val132)+(val12*val133)+(val14*val134)+(val15*val135)+(val16*val136)+(val17*val137)+(val18*val138)+(val19*val139)+(val20*val141)+(val21*val142)+(val22*val143)+(val23*val144)+(val25*val145)+(val26*val146)+(val27*val151)+(val28*val162)+(val29*val173)+(val30*val182)+(val31*val184)+(val32*val185));
    buf0[13] = (buf0[13]+(val124*val147)+(val125*val148)+(val126*val149)+(val127*val150)+(val3*val152)+(val4*val153)+(val5*val154)+(val6*val155)+(val7*val156)+(val8*val157)+(val9*val158)+(val10*val159)+(val11*val160)+(val12*val161)+(val14*val163)+(val15*val164)+(val16*val165)+(val17*val166)+(val18*val167)+(val19*val168)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val174)+(val26*val175)+(val27*val176)+(val28*val177)+(val29*val178)+(val30*val179)+(val31*val180)+(val32*val181));
    buf0[14] = (buf0[14]+(val124*val188)+(val125*val189)+(val126*val190)+(val127*val191)+(val3*val192)+(val4*val193)+(val5*val194)+(val6*val195)+(val7*val196)+(val8*val197)+(val9*val198)+(val10*val199)+(val11*val200)+(val12*val201)+(val14*val202)+(val15*val203)+(val16*val204)+(val17*val205)+(val18*val206)+(val19*val207)+(val20*val208)+(val21*val209)+(val22*val210)+(val23*val211)+(val25*val212)+(val26*val213)+(val27*val214)+(val28*val215)+(val29*val216)+(val30*val217)+(val31*val218)+(val32*val219));
    buf0[15] = (buf0[15]+(val124*val222)+(val125*val223)+(val126*val224)+(val127*val225)+(val3*val226)+(val4*val227)+(val5*val228)+(val6*val229)+(val7*val230)+(val8*val231)+(val9*val232)+(val10*val233)+(val11*val234)+(val12*val235)+(val14*val236)+(val15*val237)+(val16*val238)+(val17*val239)+(val18*val240)+(val19*val241)+(val20*val242)+(val21*val243)+(val22*val244)+(val23*val245)+(val25*val246)+(val26*val247)+(val27*val248)+(val28*val249)+(val29*val250)+(val30*val251)+(val31*val252)+(val32*val253));
    buf0[2] = (buf0[2]+(val0*val188)+(val1*val189)+(val40*val190)+(val51*val191)+(val62*val192)+(val73*val193)+(val84*val194)+(val95*val195)+(val106*val196)+(val117*val197)+(val2*val198)+(val13*val199)+(val24*val200)+(val33*val201)+(val34*val202)+(val35*val203)+(val36*val204)+(val37*val205)+(val38*val206)+(val39*val207)+(val41*val208)+(val42*val209)+(val43*val210)+(val44*val211)+(val45*val212)+(val46*val213)+(val47*val214)+(val48*val215)+(val49*val216)+(val50*val217)+(val52*val218)+(val53*val219));
    buf0[3] = (buf0[3]+(val0*val222)+(val1*val223)+(val40*val224)+(val51*val225)+(val62*val226)+(val73*val227)+(val84*val228)+(val95*val229)+(val106*val230)+(val117*val231)+(val2*val232)+(val13*val233)+(val24*val234)+(val33*val235)+(val34*val236)+(val35*val237)+(val36*val238)+(val37*val239)+(val38*val240)+(val39*val241)+(val41*val242)+(val42*val243)+(val43*val244)+(val44*val245)+(val45*val246)+(val46*val247)+(val47*val248)+(val48*val249)+(val49*val250)+(val50*val251)+(val52*val252)+(val53*val253));
    buf0[4] = (buf0[4]+(val54*val128)+(val55*val129)+(val56*val140)+(val57*val183)+(val58*val186)+(val59*val187)+(val60*val220)+(val61*val221)+(val63*val254)+(val64*val255)+(val65*val130)+(val66*val131)+(val67*val132)+(val68*val133)+(val69*val134)+(val70*val135)+(val71*val136)+(val72*val137)+(val74*val138)+(val75*val139)+(val76*val141)+(val77*val142)+(val78*val143)+(val79*val144)+(val80*val145)+(val81*val146)+(val82*val151)+(val83*val162)+(val85*val173)+(val86*val182)+(val87*val184)+(val88*val185));
    buf0[5] = (buf0[5]+(val54*val147)+(val55*val148)+(val56*val149)+(val57*val150)+(val58*val152)+(val59*val153)+(val60*val154)+(val61*val155)+(val63*val156)+(val64*val157)+(val65*val158)+(val66*val159)+(val67*val160)+(val68*val161)+(val69*val163)+(val70*val164)+(val71*val165)+(val72*val166)+(val74*val167)+(val75*val168)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val174)+(val81*val175)+(val82*val176)+(val83*val177)+(val85*val178)+(val86*val179)+(val87*val180)+(val88*val181));
    buf0[6] = (buf0[6]+(val54*val188)+(val55*val189)+(val56*val190)+(val57*val191)+(val58*val192)+(val59*val193)+(val60*val194)+(val61*val195)+(val63*val196)+(val64*val197)+(val65*val198)+(val66*val199)+(val67*val200)+(val68*val201)+(val69*val202)+(val70*val203)+(val71*val204)+(val72*val205)+(val74*val206)+(val75*val207)+(val76*val208)+(val77*val209)+(val78*val210)+(val79*val211)+(val80*val212)+(val81*val213)+(val82*val214)+(val83*val215)+(val85*val216)+(val86*val217)+(val87*val218)+(val88*val219));
    buf0[7] = (buf0[7]+(val54*val222)+(val55*val223)+(val56*val224)+(val57*val225)+(val58*val226)+(val59*val227)+(val60*val228)+(val61*val229)+(val63*val230)+(val64*val231)+(val65*val232)+(val66*val233)+(val67*val234)+(val68*val235)+(val69*val236)+(val70*val237)+(val71*val238)+(val72*val239)+(val74*val240)+(val75*val241)+(val76*val242)+(val77*val243)+(val78*val244)+(val79*val245)+(val80*val246)+(val81*val247)+(val82*val248)+(val83*val249)+(val85*val250)+(val86*val251)+(val87*val252)+(val88*val253));
    buf0[8] = (buf0[8]+(val89*val128)+(val90*val129)+(val91*val140)+(val92*val183)+(val93*val186)+(val94*val187)+(val96*val220)+(val97*val221)+(val98*val254)+(val99*val255)+(val100*val130)+(val101*val131)+(val102*val132)+(val103*val133)+(val104*val134)+(val105*val135)+(val107*val136)+(val108*val137)+(val109*val138)+(val110*val139)+(val111*val141)+(val112*val142)+(val113*val143)+(val114*val144)+(val115*val145)+(val116*val146)+(val118*val151)+(val119*val162)+(val120*val173)+(val121*val182)+(val122*val184)+(val123*val185));
    buf0[9] = (buf0[9]+(val89*val147)+(val90*val148)+(val91*val149)+(val92*val150)+(val93*val152)+(val94*val153)+(val96*val154)+(val97*val155)+(val98*val156)+(val99*val157)+(val100*val158)+(val101*val159)+(val102*val160)+(val103*val161)+(val104*val163)+(val105*val164)+(val107*val165)+(val108*val166)+(val109*val167)+(val110*val168)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val174)+(val116*val175)+(val118*val176)+(val119*val177)+(val120*val178)+(val121*val179)+(val122*val180)+(val123*val181));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<13u))+bitcast<i32>((cast2<<10u)));
  var val256 = data1_65536[alu36];
  var alu37 = (alu36+1);
  var val257 = data1_65536[alu37];
  var alu38 = (alu36+2);
  var val258 = data1_65536[alu38];
  var alu39 = (alu36+256);
  var val259 = data1_65536[alu39];
  var alu40 = (alu36+257);
  var val260 = data1_65536[alu40];
  var alu41 = (alu36+258);
  var val261 = data1_65536[alu41];
  var alu42 = (alu36+259);
  var val262 = data1_65536[alu42];
  var alu43 = (alu36+3);
  var val263 = data1_65536[alu43];
  var alu44 = (alu36+512);
  var val264 = data1_65536[alu44];
  var alu45 = (alu36+513);
  var val265 = data1_65536[alu45];
  var alu46 = (alu36+514);
  var val266 = data1_65536[alu46];
  var alu47 = (alu36+515);
  var val267 = data1_65536[alu47];
  var alu48 = (alu36+768);
  var val268 = data1_65536[alu48];
  var alu49 = (alu36+769);
  var val269 = data1_65536[alu49];
  var alu50 = (alu36+770);
  var val270 = data1_65536[alu50];
  var alu51 = (alu36+771);
  var val271 = data1_65536[alu51];
  var val272 = data2_1536[(alu35+512)];
  var val273 = data2_1536[(alu35+513)];
  var val274 = data2_1536[(alu35+514)];
  var val275 = data2_1536[(alu35+515)];
  var val276 = data5_256[alu35];
  var val277 = data5_256[(alu35+1)];
  var val278 = data5_256[(alu35+2)];
  var val279 = data5_256[(alu35+3)];
  data0_65536[alu36] = (val256+(val272*(buf0[0]+val276)));
  data0_65536[alu37] = (val257+(val273*(buf0[1]+val277)));
  data0_65536[alu38] = (val258+(val274*(buf0[2]+val278)));
  data0_65536[alu39] = (val259+(val272*(buf0[4]+val276)));
  data0_65536[alu40] = (val260+(val273*(buf0[5]+val277)));
  data0_65536[alu41] = (val261+(val274*(buf0[6]+val278)));
  data0_65536[alu42] = (val262+(val275*(buf0[7]+val279)));
  data0_65536[alu43] = (val263+(val275*(buf0[3]+val279)));
  data0_65536[alu44] = (val264+(val272*(buf0[8]+val276)));
  data0_65536[alu45] = (val265+(val273*(buf0[9]+val277)));
  data0_65536[alu46] = (val266+(val274*(buf0[10]+val278)));
  data0_65536[alu47] = (val267+(val275*(buf0[11]+val279)));
  data0_65536[alu48] = (val268+(val272*(buf0[12]+val276)));
  data0_65536[alu49] = (val269+(val273*(buf0[13]+val277)));
  data0_65536[alu50] = (val270+(val274*(buf0[14]+val278)));
  data0_65536[alu51] = (val271+(val275*(buf0[15]+val279)));
}`;

const r_16_8_4_4_32_dispatch_73 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_74 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var val21 = data3_256[(alu0+1)];
  var val22 = data3_256[(alu0+2)];
  var val23 = data3_256[(alu0+3)];
  var val24 = data4_1536[(alu0+1024)];
  var val25 = data4_1536[(alu0+1025)];
  var val26 = data4_1536[(alu0+1026)];
  var val27 = data4_1536[(alu0+1027)];
  var val28 = data4_1536[(alu0+768)];
  var val29 = data4_1536[(alu0+769)];
  var val30 = data4_1536[(alu0+770)];
  var val31 = data4_1536[(alu0+771)];
  var alu18 = (1/val16);
  var alu19 = (1/val17);
  var alu20 = (1/val18);
  var alu21 = (1/val19);
  var alu22 = (1.0f+val24);
  var alu23 = (1.0f+val25);
  var alu24 = (1.0f+val26);
  var alu25 = (1.0f+val27);
  data0_65536[alu1] = ((val0*alu18*val20*alu22)+val28);
  data0_65536[alu2] = ((val1*alu18*val21*alu23)+val29);
  data0_65536[alu3] = ((val2*alu18*val22*alu24)+val30);
  data0_65536[alu4] = ((val3*alu19*val20*alu22)+val28);
  data0_65536[alu5] = ((val4*alu19*val21*alu23)+val29);
  data0_65536[alu6] = ((val5*alu19*val22*alu24)+val30);
  data0_65536[alu7] = ((val6*alu19*val23*alu25)+val31);
  data0_65536[alu8] = ((val7*alu18*val23*alu25)+val31);
  data0_65536[alu9] = ((val8*alu20*val20*alu22)+val28);
  data0_65536[alu10] = ((val9*alu20*val21*alu23)+val29);
  data0_65536[alu11] = ((val10*alu20*val22*alu24)+val30);
  data0_65536[alu12] = ((val11*alu20*val23*alu25)+val31);
  data0_65536[alu13] = ((val12*alu21*val20*alu22)+val28);
  data0_65536[alu14] = ((val13*alu21*val21*alu23)+val29);
  data0_65536[alu15] = ((val14*alu21*val22*alu24)+val30);
  data0_65536[alu16] = ((val15*alu21*val23*alu25)+val31);
}`;

const r_8_22_8_16_4_4_64_4_dispatch_75 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_360448:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_360448:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1408:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 22 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast2 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast2);
    var val0 = data1_65536[alu16];
    var val1 = data1_65536[(alu16+1)];
    var val2 = data1_65536[(alu16+2)];
    var val3 = data1_65536[(alu16+256)];
    var val4 = data1_65536[(alu16+257)];
    var val5 = data1_65536[(alu16+258)];
    var val6 = data1_65536[(alu16+259)];
    var val7 = data1_65536[(alu16+3)];
    var val8 = data1_65536[(alu16+512)];
    var val9 = data1_65536[(alu16+513)];
    var val10 = data1_65536[(alu16+514)];
    var val11 = data1_65536[(alu16+515)];
    var val12 = data1_65536[(alu16+768)];
    var val13 = data1_65536[(alu16+769)];
    var val14 = data1_65536[(alu16+770)];
    var val15 = data1_65536[(alu16+771)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast1<<10u))+cast2);
    var val16 = data2_360448[alu17];
    var val17 = data2_360448[(alu17+1)];
    var val18 = data2_360448[(alu17+2)];
    var val19 = data2_360448[(alu17+256)];
    var val20 = data2_360448[(alu17+257)];
    var val21 = data2_360448[(alu17+258)];
    var val22 = data2_360448[(alu17+259)];
    var val23 = data2_360448[(alu17+3)];
    var val24 = data2_360448[(alu17+512)];
    var val25 = data2_360448[(alu17+513)];
    var val26 = data2_360448[(alu17+514)];
    var val27 = data2_360448[(alu17+515)];
    var val28 = data2_360448[(alu17+768)];
    var val29 = data2_360448[(alu17+769)];
    var val30 = data2_360448[(alu17+770)];
    var val31 = data2_360448[(alu17+771)];
    buf0[0] = (buf0[0]+(val0*val16)+(val1*val17)+(val2*val18)+(val7*val23));
    buf0[1] = (buf0[1]+(val0*val19)+(val1*val20)+(val2*val21)+(val7*val22));
    buf0[10] = (buf0[10]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    buf0[11] = (buf0[11]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    buf0[12] = (buf0[12]+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val23));
    buf0[13] = (buf0[13]+(val12*val19)+(val13*val20)+(val14*val21)+(val15*val22));
    buf0[14] = (buf0[14]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    buf0[15] = (buf0[15]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    buf0[2] = (buf0[2]+(val0*val24)+(val1*val25)+(val2*val26)+(val7*val27));
    buf0[3] = (buf0[3]+(val0*val28)+(val1*val29)+(val2*val30)+(val7*val31));
    buf0[4] = (buf0[4]+(val3*val16)+(val4*val17)+(val5*val18)+(val6*val23));
    buf0[5] = (buf0[5]+(val3*val19)+(val4*val20)+(val5*val21)+(val6*val22));
    buf0[6] = (buf0[6]+(val3*val24)+(val4*val25)+(val5*val26)+(val6*val27));
    buf0[7] = (buf0[7]+(val3*val28)+(val4*val29)+(val5*val30)+(val6*val31));
    buf0[8] = (buf0[8]+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val23));
    buf0[9] = (buf0[9]+(val8*val19)+(val9*val20)+(val10*val21)+(val11*val22));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)));
  var val32 = data3_1408[alu35];
  var val33 = data3_1408[(alu35+1)];
  var val34 = data3_1408[(alu35+2)];
  var val35 = data3_1408[(alu35+3)];
  var alu36 = (alu35+(gidx1*45056)+(lidx0*5632));
  data0_360448[alu36] = (buf0[0]+val32);
  data0_360448[(alu36+1)] = (buf0[1]+val33);
  data0_360448[(alu36+1408)] = (buf0[4]+val32);
  data0_360448[(alu36+1409)] = (buf0[5]+val33);
  data0_360448[(alu36+1410)] = (buf0[6]+val34);
  data0_360448[(alu36+1411)] = (buf0[7]+val35);
  data0_360448[(alu36+2)] = (buf0[2]+val34);
  data0_360448[(alu36+2816)] = (buf0[8]+val32);
  data0_360448[(alu36+2817)] = (buf0[9]+val33);
  data0_360448[(alu36+2818)] = (buf0[10]+val34);
  data0_360448[(alu36+2819)] = (buf0[11]+val35);
  data0_360448[(alu36+3)] = (buf0[3]+val35);
  data0_360448[(alu36+4224)] = (buf0[12]+val32);
  data0_360448[(alu36+4225)] = (buf0[13]+val33);
  data0_360448[(alu36+4226)] = (buf0[14]+val34);
  data0_360448[(alu36+4227)] = (buf0[15]+val35);
}`;

const r_8_4_8_16_4_4_176_4_dispatch_76 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_360448:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_180224:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 176; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = ((gidx1*45056)+(lidx0*5632)+cast0);
    var val0 = data3_360448[(alu16+1)];
    var val1 = data3_360448[(alu16+1408)];
    var val2 = data3_360448[(alu16+1409)];
    var val3 = data3_360448[(alu16+1410)];
    var val4 = data3_360448[(alu16+1411)];
    var val5 = data3_360448[(alu16+2)];
    var val6 = data3_360448[(alu16+2112)];
    var val7 = data3_360448[(alu16+2113)];
    var val8 = data3_360448[(alu16+2114)];
    var val9 = data3_360448[(alu16+2115)];
    var val10 = data3_360448[(alu16+2816)];
    var val11 = data3_360448[(alu16+2817)];
    var val12 = data3_360448[(alu16+2818)];
    var val13 = data3_360448[(alu16+2819)];
    var val14 = data3_360448[(alu16+3)];
    var val15 = data3_360448[(alu16+3520)];
    var val16 = data3_360448[(alu16+3521)];
    var val17 = data3_360448[(alu16+3522)];
    var val18 = data3_360448[(alu16+3523)];
    var val19 = data3_360448[(alu16+4224)];
    var val20 = data3_360448[(alu16+4225)];
    var val21 = data3_360448[(alu16+4226)];
    var val22 = data3_360448[(alu16+4227)];
    var val23 = data3_360448[(alu16+4928)];
    var val24 = data3_360448[(alu16+4929)];
    var val25 = data3_360448[(alu16+4930)];
    var val26 = data3_360448[(alu16+4931)];
    var val27 = data3_360448[(alu16+704)];
    var val28 = data3_360448[(alu16+705)];
    var val29 = data3_360448[(alu16+706)];
    var val30 = data3_360448[(alu16+707)];
    var val31 = data3_360448[alu16];
    var alu17 = ((gidx0*45056)+(lidx1*2816)+cast0);
    var val32 = data4_180224[(alu17+1)];
    var val33 = data4_180224[(alu17+1408)];
    var val34 = data4_180224[(alu17+1409)];
    var val35 = data4_180224[(alu17+1410)];
    var val36 = data4_180224[(alu17+1411)];
    var val37 = data4_180224[(alu17+2)];
    var val38 = data4_180224[(alu17+2112)];
    var val39 = data4_180224[(alu17+2113)];
    var val40 = data4_180224[(alu17+2114)];
    var val41 = data4_180224[(alu17+2115)];
    var val42 = data4_180224[(alu17+3)];
    var val43 = data4_180224[(alu17+704)];
    var val44 = data4_180224[(alu17+705)];
    var val45 = data4_180224[(alu17+706)];
    var val46 = data4_180224[(alu17+707)];
    var val47 = data4_180224[alu17];
    var alu18 = (val0*(1/(1.0f+exp2((val0*-1.4426950216293335f))))*val28);
    var alu19 = (val31*(1/(1.0f+exp2((val31*-1.4426950216293335f))))*val27);
    var alu20 = (val5*(1/(1.0f+exp2((val5*-1.4426950216293335f))))*val29);
    var alu21 = (val14*(1/(1.0f+exp2((val14*-1.4426950216293335f))))*val30);
    var alu22 = (val10*(1/(1.0f+exp2((val10*-1.4426950216293335f))))*val15);
    var alu23 = (val11*(1/(1.0f+exp2((val11*-1.4426950216293335f))))*val16);
    var alu24 = (val12*(1/(1.0f+exp2((val12*-1.4426950216293335f))))*val17);
    var alu25 = (val13*(1/(1.0f+exp2((val13*-1.4426950216293335f))))*val18);
    var alu26 = (val19*(1/(1.0f+exp2((val19*-1.4426950216293335f))))*val23);
    var alu27 = (val20*(1/(1.0f+exp2((val20*-1.4426950216293335f))))*val24);
    var alu28 = (val21*(1/(1.0f+exp2((val21*-1.4426950216293335f))))*val25);
    var alu29 = (val22*(1/(1.0f+exp2((val22*-1.4426950216293335f))))*val26);
    var alu30 = (val1*(1/(1.0f+exp2((val1*-1.4426950216293335f))))*val6);
    var alu31 = (val2*(1/(1.0f+exp2((val2*-1.4426950216293335f))))*val7);
    var alu32 = (val3*(1/(1.0f+exp2((val3*-1.4426950216293335f))))*val8);
    var alu33 = (val4*(1/(1.0f+exp2((val4*-1.4426950216293335f))))*val9);
    buf0[0] = (buf0[0]+(alu19*val47)+(alu18*val32)+(alu20*val37)+(alu21*val42));
    buf0[1] = (buf0[1]+(alu19*val43)+(alu18*val44)+(alu20*val45)+(alu21*val46));
    buf0[10] = (buf0[10]+(alu22*val33)+(alu23*val34)+(alu24*val35)+(alu25*val36));
    buf0[11] = (buf0[11]+(alu22*val38)+(alu23*val39)+(alu24*val40)+(alu25*val41));
    buf0[12] = (buf0[12]+(alu26*val47)+(alu27*val32)+(alu28*val37)+(alu29*val42));
    buf0[13] = (buf0[13]+(alu26*val43)+(alu27*val44)+(alu28*val45)+(alu29*val46));
    buf0[14] = (buf0[14]+(alu26*val33)+(alu27*val34)+(alu28*val35)+(alu29*val36));
    buf0[15] = (buf0[15]+(alu26*val38)+(alu27*val39)+(alu28*val40)+(alu29*val41));
    buf0[2] = (buf0[2]+(alu19*val33)+(alu18*val34)+(alu20*val35)+(alu21*val36));
    buf0[3] = (buf0[3]+(alu19*val38)+(alu18*val39)+(alu20*val40)+(alu21*val41));
    buf0[4] = (buf0[4]+(alu30*val47)+(alu31*val32)+(alu32*val37)+(alu33*val42));
    buf0[5] = (buf0[5]+(alu30*val43)+(alu31*val44)+(alu32*val45)+(alu33*val46));
    buf0[6] = (buf0[6]+(alu30*val33)+(alu31*val34)+(alu32*val35)+(alu33*val36));
    buf0[7] = (buf0[7]+(alu30*val38)+(alu31*val39)+(alu32*val40)+(alu33*val41));
    buf0[8] = (buf0[8]+(alu22*val47)+(alu23*val32)+(alu24*val37)+(alu25*val42));
    buf0[9] = (buf0[9]+(alu22*val43)+(alu23*val44)+(alu24*val45)+(alu25*val46));
  }
  var alu51 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu52 = (alu51+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u)));
  var val48 = data1_65536[alu52];
  var alu53 = (alu52+1);
  var val49 = data1_65536[alu53];
  var alu54 = (alu52+2);
  var val50 = data1_65536[alu54];
  var alu55 = (alu52+256);
  var val51 = data1_65536[alu55];
  var alu56 = (alu52+257);
  var val52 = data1_65536[alu56];
  var alu57 = (alu52+258);
  var val53 = data1_65536[alu57];
  var alu58 = (alu52+259);
  var val54 = data1_65536[alu58];
  var alu59 = (alu52+3);
  var val55 = data1_65536[alu59];
  var alu60 = (alu52+512);
  var val56 = data1_65536[alu60];
  var alu61 = (alu52+513);
  var val57 = data1_65536[alu61];
  var alu62 = (alu52+514);
  var val58 = data1_65536[alu62];
  var alu63 = (alu52+515);
  var val59 = data1_65536[alu63];
  var alu64 = (alu52+768);
  var val60 = data1_65536[alu64];
  var alu65 = (alu52+769);
  var val61 = data1_65536[alu65];
  var alu66 = (alu52+770);
  var val62 = data1_65536[alu66];
  var alu67 = (alu52+771);
  var val63 = data1_65536[alu67];
  var val64 = data2_1536[(alu51+1280)];
  var val65 = data2_1536[(alu51+1281)];
  var val66 = data2_1536[(alu51+1282)];
  var val67 = data2_1536[(alu51+1283)];
  var val68 = data5_256[alu51];
  var val69 = data5_256[(alu51+1)];
  var val70 = data5_256[(alu51+2)];
  var val71 = data5_256[(alu51+3)];
  data0_65536[alu52] = (val48+(val64*(buf0[0]+val68)));
  data0_65536[alu53] = (val49+(val65*(buf0[1]+val69)));
  data0_65536[alu54] = (val50+(val66*(buf0[2]+val70)));
  data0_65536[alu55] = (val51+(val64*(buf0[4]+val68)));
  data0_65536[alu56] = (val52+(val65*(buf0[5]+val69)));
  data0_65536[alu57] = (val53+(val66*(buf0[6]+val70)));
  data0_65536[alu58] = (val54+(val67*(buf0[7]+val71)));
  data0_65536[alu59] = (val55+(val67*(buf0[3]+val71)));
  data0_65536[alu60] = (val56+(val64*(buf0[8]+val68)));
  data0_65536[alu61] = (val57+(val65*(buf0[9]+val69)));
  data0_65536[alu62] = (val58+(val66*(buf0[10]+val70)));
  data0_65536[alu63] = (val59+(val67*(buf0[11]+val71)));
  data0_65536[alu64] = (val60+(val64*(buf0[12]+val68)));
  data0_65536[alu65] = (val61+(val65*(buf0[13]+val69)));
  data0_65536[alu66] = (val62+(val66*(buf0[14]+val70)));
  data0_65536[alu67] = (val63+(val67*(buf0[15]+val71)));
}`;

const r_16_8_4_4_32_dispatch_77 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_78 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var alu18 = (alu0+1);
  var val21 = data3_256[alu18];
  var alu19 = (alu0+2);
  var val22 = data3_256[alu19];
  var alu20 = (alu0+3);
  var val23 = data3_256[alu20];
  var val24 = data4_1536[alu0];
  var val25 = data4_1536[alu18];
  var val26 = data4_1536[alu19];
  var val27 = data4_1536[(alu0+256)];
  var val28 = data4_1536[(alu0+257)];
  var val29 = data4_1536[(alu0+258)];
  var val30 = data4_1536[(alu0+259)];
  var val31 = data4_1536[alu20];
  var alu21 = (1/val16);
  var alu22 = (1/val17);
  var alu23 = (1/val18);
  var alu24 = (1/val19);
  var alu25 = (1.0f+val27);
  var alu26 = (1.0f+val28);
  var alu27 = (1.0f+val29);
  var alu28 = (1.0f+val30);
  data0_65536[alu1] = ((val0*alu21*val20*alu25)+val24);
  data0_65536[alu2] = ((val1*alu21*val21*alu26)+val25);
  data0_65536[alu3] = ((val2*alu21*val22*alu27)+val26);
  data0_65536[alu4] = ((val3*alu22*val20*alu25)+val24);
  data0_65536[alu5] = ((val4*alu22*val21*alu26)+val25);
  data0_65536[alu6] = ((val5*alu22*val22*alu27)+val26);
  data0_65536[alu7] = ((val6*alu22*val23*alu28)+val31);
  data0_65536[alu8] = ((val7*alu21*val23*alu28)+val31);
  data0_65536[alu9] = ((val8*alu23*val20*alu25)+val24);
  data0_65536[alu10] = ((val9*alu23*val21*alu26)+val25);
  data0_65536[alu11] = ((val10*alu23*val22*alu27)+val26);
  data0_65536[alu12] = ((val11*alu23*val23*alu28)+val31);
  data0_65536[alu13] = ((val12*alu24*val20*alu25)+val24);
  data0_65536[alu14] = ((val13*alu24*val21*alu26)+val25);
  data0_65536[alu15] = ((val14*alu24*val22*alu27)+val26);
  data0_65536[alu16] = ((val15*alu24*val23*alu28)+val31);
}`;

const r_8_16_8_16_3_4_64_4_dispatch_79 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_196608:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu12 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast0);
    var val0 = data1_65536[alu12];
    var val1 = data1_65536[(alu12+1)];
    var val2 = data1_65536[(alu12+2)];
    var val3 = data1_65536[(alu12+256)];
    var val4 = data1_65536[(alu12+257)];
    var val5 = data1_65536[(alu12+258)];
    var val6 = data1_65536[(alu12+259)];
    var val7 = data1_65536[(alu12+3)];
    var val8 = data1_65536[(alu12+512)];
    var val9 = data1_65536[(alu12+513)];
    var val10 = data1_65536[(alu12+514)];
    var val11 = data1_65536[(alu12+515)];
    var val12 = data1_65536[(alu12+768)];
    var val13 = data1_65536[(alu12+769)];
    var val14 = data1_65536[(alu12+770)];
    var val15 = data1_65536[(alu12+771)];
    var alu13 = ((gidx0*12288)+(lidx1*768)+cast0);
    var val16 = data2_196608[(alu13+1)];
    var val17 = data2_196608[(alu13+2)];
    var val18 = data2_196608[(alu13+256)];
    var val19 = data2_196608[(alu13+257)];
    var val20 = data2_196608[(alu13+258)];
    var val21 = data2_196608[(alu13+259)];
    var val22 = data2_196608[(alu13+3)];
    var val23 = data2_196608[(alu13+512)];
    var val24 = data2_196608[(alu13+513)];
    var val25 = data2_196608[(alu13+514)];
    var val26 = data2_196608[(alu13+515)];
    var val27 = data2_196608[alu13];
    buf0[0] = (buf0[0]+(val0*val27)+(val1*val16)+(val2*val17)+(val7*val22));
    buf0[1] = (buf0[1]+(val0*val18)+(val1*val19)+(val2*val20)+(val7*val21));
    buf0[10] = (buf0[10]+(val12*val18)+(val13*val19)+(val14*val20)+(val15*val21));
    buf0[11] = (buf0[11]+(val12*val23)+(val13*val24)+(val14*val25)+(val15*val26));
    buf0[2] = (buf0[2]+(val0*val23)+(val1*val24)+(val2*val25)+(val7*val26));
    buf0[3] = (buf0[3]+(val3*val27)+(val4*val16)+(val5*val17)+(val6*val22));
    buf0[4] = (buf0[4]+(val3*val18)+(val4*val19)+(val5*val20)+(val6*val21));
    buf0[5] = (buf0[5]+(val3*val23)+(val4*val24)+(val5*val25)+(val6*val26));
    buf0[6] = (buf0[6]+(val8*val27)+(val9*val16)+(val10*val17)+(val11*val22));
    buf0[7] = (buf0[7]+(val8*val18)+(val9*val19)+(val10*val20)+(val11*val21));
    buf0[8] = (buf0[8]+(val8*val23)+(val9*val24)+(val10*val25)+(val11*val26));
    buf0[9] = (buf0[9]+(val12*val27)+(val13*val16)+(val14*val17)+(val15*val22));
  }
  var alu27 = ((gidx0*48)+(lidx1*3));
  var val28 = data3_768[(alu27+1)];
  var val29 = data3_768[(alu27+2)];
  var val30 = data3_768[alu27];
  var alu28 = (alu27+(gidx1*24576)+(lidx0*3072));
  data0_196608[(alu28+1)] = (buf0[1]+val28);
  data0_196608[(alu28+1536)] = (buf0[6]+val30);
  data0_196608[(alu28+1537)] = (buf0[7]+val28);
  data0_196608[(alu28+1538)] = (buf0[8]+val29);
  data0_196608[(alu28+2)] = (buf0[2]+val29);
  data0_196608[(alu28+2304)] = (buf0[9]+val30);
  data0_196608[(alu28+2305)] = (buf0[10]+val28);
  data0_196608[(alu28+2306)] = (buf0[11]+val29);
  data0_196608[(alu28+768)] = (buf0[3]+val30);
  data0_196608[(alu28+769)] = (buf0[4]+val28);
  data0_196608[(alu28+770)] = (buf0[5]+val29);
  data0_196608[alu28] = (buf0[0]+val30);
}`;

const r_8_16_8_4_4_4_dispatch_80 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[alu4];
    var val1 = data1_196608[(alu4+3072)];
    var val2 = data1_196608[(alu4+6144)];
    var val3 = data1_196608[(alu4+9216)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const r_8_16_8_4_4_4_dispatch_81 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[(alu4+256)];
    var val1 = data1_196608[(alu4+3328)];
    var val2 = data1_196608[(alu4+6400)];
    var val3 = data1_196608[(alu4+9472)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const E_2_32_8_16_2_4_dispatch_82 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+1)];
  var val1 = data1_196608[(alu0+32)];
  var val2 = data1_196608[(alu0+33)];
  var val3 = data1_196608[(alu0+64)];
  var val4 = data1_196608[(alu0+65)];
  var val5 = data1_196608[(alu0+96)];
  var val6 = data1_196608[(alu0+97)];
  var val7 = data1_196608[alu0];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val13);
  var alu10 = (val7*alu5*val12);
  var alu11 = (val1*alu6*val12);
  var alu12 = (val2*alu6*val13);
  var alu13 = (val3*alu7*val12);
  var alu14 = (val4*alu7*val13);
  var alu15 = (val5*alu8*val12);
  var alu16 = (val6*alu8*val13);
  data0_65536[alu4] = ((alu10*val14)-(alu9*val16));
  data0_65536[(alu4+1)] = ((alu9*val15)+(alu10*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const E_2_32_8_16_2_4_dispatch_83 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+256)];
  var val1 = data1_196608[(alu0+257)];
  var val2 = data1_196608[(alu0+288)];
  var val3 = data1_196608[(alu0+289)];
  var val4 = data1_196608[(alu0+320)];
  var val5 = data1_196608[(alu0+321)];
  var val6 = data1_196608[(alu0+352)];
  var val7 = data1_196608[(alu0+353)];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val12);
  var alu10 = (val1*alu5*val13);
  var alu11 = (val2*alu6*val12);
  var alu12 = (val3*alu6*val13);
  var alu13 = (val4*alu7*val12);
  var alu14 = (val5*alu7*val13);
  var alu15 = (val6*alu8*val12);
  var alu16 = (val7*alu8*val13);
  data0_65536[alu4] = ((alu9*val14)-(alu10*val16));
  data0_65536[(alu4+1)] = ((alu10*val15)+(alu9*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const r_8_8_4_8_16_4_4_32_dispatch_84 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_524288:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
  var cast2 = bitcast<i32>((cast1<<13u));
  var cast3 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<10u))+bitcast<i32>((cast3<<7u))+cast2);
  var val0 = data1_65536[alu0];
  var val1 = data1_65536[(alu0+1)];
  var val2 = data1_65536[(alu0+10)];
  var val3 = data1_65536[(alu0+100)];
  var val4 = data1_65536[(alu0+101)];
  var val5 = data1_65536[(alu0+102)];
  var val6 = data1_65536[(alu0+103)];
  var val7 = data1_65536[(alu0+104)];
  var val8 = data1_65536[(alu0+105)];
  var val9 = data1_65536[(alu0+106)];
  var val10 = data1_65536[(alu0+107)];
  var val11 = data1_65536[(alu0+108)];
  var val12 = data1_65536[(alu0+109)];
  var val13 = data1_65536[(alu0+11)];
  var val14 = data1_65536[(alu0+110)];
  var val15 = data1_65536[(alu0+111)];
  var val16 = data1_65536[(alu0+112)];
  var val17 = data1_65536[(alu0+113)];
  var val18 = data1_65536[(alu0+114)];
  var val19 = data1_65536[(alu0+115)];
  var val20 = data1_65536[(alu0+116)];
  var val21 = data1_65536[(alu0+117)];
  var val22 = data1_65536[(alu0+118)];
  var val23 = data1_65536[(alu0+119)];
  var val24 = data1_65536[(alu0+12)];
  var val25 = data1_65536[(alu0+120)];
  var val26 = data1_65536[(alu0+121)];
  var val27 = data1_65536[(alu0+122)];
  var val28 = data1_65536[(alu0+123)];
  var val29 = data1_65536[(alu0+124)];
  var val30 = data1_65536[(alu0+125)];
  var val31 = data1_65536[(alu0+126)];
  var val32 = data1_65536[(alu0+127)];
  var val33 = data1_65536[(alu0+13)];
  var val34 = data1_65536[(alu0+14)];
  var val35 = data1_65536[(alu0+15)];
  var val36 = data1_65536[(alu0+16)];
  var val37 = data1_65536[(alu0+17)];
  var val38 = data1_65536[(alu0+18)];
  var val39 = data1_65536[(alu0+19)];
  var val40 = data1_65536[(alu0+2)];
  var val41 = data1_65536[(alu0+20)];
  var val42 = data1_65536[(alu0+21)];
  var val43 = data1_65536[(alu0+22)];
  var val44 = data1_65536[(alu0+23)];
  var val45 = data1_65536[(alu0+24)];
  var val46 = data1_65536[(alu0+25)];
  var val47 = data1_65536[(alu0+26)];
  var val48 = data1_65536[(alu0+27)];
  var val49 = data1_65536[(alu0+28)];
  var val50 = data1_65536[(alu0+29)];
  var val51 = data1_65536[(alu0+3)];
  var val52 = data1_65536[(alu0+30)];
  var val53 = data1_65536[(alu0+31)];
  var val54 = data1_65536[(alu0+32)];
  var val55 = data1_65536[(alu0+33)];
  var val56 = data1_65536[(alu0+34)];
  var val57 = data1_65536[(alu0+35)];
  var val58 = data1_65536[(alu0+36)];
  var val59 = data1_65536[(alu0+37)];
  var val60 = data1_65536[(alu0+38)];
  var val61 = data1_65536[(alu0+39)];
  var val62 = data1_65536[(alu0+4)];
  var val63 = data1_65536[(alu0+40)];
  var val64 = data1_65536[(alu0+41)];
  var val65 = data1_65536[(alu0+42)];
  var val66 = data1_65536[(alu0+43)];
  var val67 = data1_65536[(alu0+44)];
  var val68 = data1_65536[(alu0+45)];
  var val69 = data1_65536[(alu0+46)];
  var val70 = data1_65536[(alu0+47)];
  var val71 = data1_65536[(alu0+48)];
  var val72 = data1_65536[(alu0+49)];
  var val73 = data1_65536[(alu0+5)];
  var val74 = data1_65536[(alu0+50)];
  var val75 = data1_65536[(alu0+51)];
  var val76 = data1_65536[(alu0+52)];
  var val77 = data1_65536[(alu0+53)];
  var val78 = data1_65536[(alu0+54)];
  var val79 = data1_65536[(alu0+55)];
  var val80 = data1_65536[(alu0+56)];
  var val81 = data1_65536[(alu0+57)];
  var val82 = data1_65536[(alu0+58)];
  var val83 = data1_65536[(alu0+59)];
  var val84 = data1_65536[(alu0+6)];
  var val85 = data1_65536[(alu0+60)];
  var val86 = data1_65536[(alu0+61)];
  var val87 = data1_65536[(alu0+62)];
  var val88 = data1_65536[(alu0+63)];
  var val89 = data1_65536[(alu0+64)];
  var val90 = data1_65536[(alu0+65)];
  var val91 = data1_65536[(alu0+66)];
  var val92 = data1_65536[(alu0+67)];
  var val93 = data1_65536[(alu0+68)];
  var val94 = data1_65536[(alu0+69)];
  var val95 = data1_65536[(alu0+7)];
  var val96 = data1_65536[(alu0+70)];
  var val97 = data1_65536[(alu0+71)];
  var val98 = data1_65536[(alu0+72)];
  var val99 = data1_65536[(alu0+73)];
  var val100 = data1_65536[(alu0+74)];
  var val101 = data1_65536[(alu0+75)];
  var val102 = data1_65536[(alu0+76)];
  var val103 = data1_65536[(alu0+77)];
  var val104 = data1_65536[(alu0+78)];
  var val105 = data1_65536[(alu0+79)];
  var val106 = data1_65536[(alu0+8)];
  var val107 = data1_65536[(alu0+80)];
  var val108 = data1_65536[(alu0+81)];
  var val109 = data1_65536[(alu0+82)];
  var val110 = data1_65536[(alu0+83)];
  var val111 = data1_65536[(alu0+84)];
  var val112 = data1_65536[(alu0+85)];
  var val113 = data1_65536[(alu0+86)];
  var val114 = data1_65536[(alu0+87)];
  var val115 = data1_65536[(alu0+88)];
  var val116 = data1_65536[(alu0+89)];
  var val117 = data1_65536[(alu0+9)];
  var val118 = data1_65536[(alu0+90)];
  var val119 = data1_65536[(alu0+91)];
  var val120 = data1_65536[(alu0+92)];
  var val121 = data1_65536[(alu0+93)];
  var val122 = data1_65536[(alu0+94)];
  var val123 = data1_65536[(alu0+95)];
  var val124 = data1_65536[(alu0+96)];
  var val125 = data1_65536[(alu0+97)];
  var val126 = data1_65536[(alu0+98)];
  var val127 = data1_65536[(alu0+99)];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast4 = bitcast<u32>(gidx0);
  var cast5 = bitcast<u32>(lidx1);
  var alu1 = (bitcast<i32>((cast4<<11u))+bitcast<i32>((cast5<<7u))+cast2);
  var val128 = data2_65536[alu1];
  var val129 = data2_65536[(alu1+1)];
  var val130 = data2_65536[(alu1+10)];
  var val131 = data2_65536[(alu1+100)];
  var val132 = data2_65536[(alu1+101)];
  var val133 = data2_65536[(alu1+102)];
  var val134 = data2_65536[(alu1+103)];
  var val135 = data2_65536[(alu1+104)];
  var val136 = data2_65536[(alu1+105)];
  var val137 = data2_65536[(alu1+106)];
  var val138 = data2_65536[(alu1+107)];
  var val139 = data2_65536[(alu1+108)];
  var val140 = data2_65536[(alu1+109)];
  var val141 = data2_65536[(alu1+11)];
  var val142 = data2_65536[(alu1+110)];
  var val143 = data2_65536[(alu1+111)];
  var val144 = data2_65536[(alu1+112)];
  var val145 = data2_65536[(alu1+113)];
  var val146 = data2_65536[(alu1+114)];
  var val147 = data2_65536[(alu1+115)];
  var val148 = data2_65536[(alu1+116)];
  var val149 = data2_65536[(alu1+117)];
  var val150 = data2_65536[(alu1+118)];
  var val151 = data2_65536[(alu1+119)];
  var val152 = data2_65536[(alu1+12)];
  var val153 = data2_65536[(alu1+120)];
  var val154 = data2_65536[(alu1+121)];
  var val155 = data2_65536[(alu1+122)];
  var val156 = data2_65536[(alu1+123)];
  var val157 = data2_65536[(alu1+124)];
  var val158 = data2_65536[(alu1+125)];
  var val159 = data2_65536[(alu1+126)];
  var val160 = data2_65536[(alu1+127)];
  var val161 = data2_65536[(alu1+13)];
  var val162 = data2_65536[(alu1+14)];
  var val163 = data2_65536[(alu1+15)];
  var val164 = data2_65536[(alu1+16)];
  var val165 = data2_65536[(alu1+17)];
  var val166 = data2_65536[(alu1+18)];
  var val167 = data2_65536[(alu1+19)];
  var val168 = data2_65536[(alu1+2)];
  var val169 = data2_65536[(alu1+20)];
  var val170 = data2_65536[(alu1+21)];
  var val171 = data2_65536[(alu1+22)];
  var val172 = data2_65536[(alu1+23)];
  var val173 = data2_65536[(alu1+24)];
  var val174 = data2_65536[(alu1+25)];
  var val175 = data2_65536[(alu1+26)];
  var val176 = data2_65536[(alu1+27)];
  var val177 = data2_65536[(alu1+28)];
  var val178 = data2_65536[(alu1+29)];
  var val179 = data2_65536[(alu1+3)];
  var val180 = data2_65536[(alu1+30)];
  var val181 = data2_65536[(alu1+31)];
  var val182 = data2_65536[(alu1+32)];
  var val183 = data2_65536[(alu1+33)];
  var val184 = data2_65536[(alu1+34)];
  var val185 = data2_65536[(alu1+35)];
  var val186 = data2_65536[(alu1+36)];
  var val187 = data2_65536[(alu1+37)];
  var val188 = data2_65536[(alu1+38)];
  var val189 = data2_65536[(alu1+39)];
  var val190 = data2_65536[(alu1+4)];
  var val191 = data2_65536[(alu1+40)];
  var val192 = data2_65536[(alu1+41)];
  var val193 = data2_65536[(alu1+42)];
  var val194 = data2_65536[(alu1+43)];
  var val195 = data2_65536[(alu1+44)];
  var val196 = data2_65536[(alu1+45)];
  var val197 = data2_65536[(alu1+46)];
  var val198 = data2_65536[(alu1+47)];
  var val199 = data2_65536[(alu1+48)];
  var val200 = data2_65536[(alu1+49)];
  var val201 = data2_65536[(alu1+5)];
  var val202 = data2_65536[(alu1+50)];
  var val203 = data2_65536[(alu1+51)];
  var val204 = data2_65536[(alu1+52)];
  var val205 = data2_65536[(alu1+53)];
  var val206 = data2_65536[(alu1+54)];
  var val207 = data2_65536[(alu1+55)];
  var val208 = data2_65536[(alu1+56)];
  var val209 = data2_65536[(alu1+57)];
  var val210 = data2_65536[(alu1+58)];
  var val211 = data2_65536[(alu1+59)];
  var val212 = data2_65536[(alu1+6)];
  var val213 = data2_65536[(alu1+60)];
  var val214 = data2_65536[(alu1+61)];
  var val215 = data2_65536[(alu1+62)];
  var val216 = data2_65536[(alu1+63)];
  var val217 = data2_65536[(alu1+64)];
  var val218 = data2_65536[(alu1+65)];
  var val219 = data2_65536[(alu1+66)];
  var val220 = data2_65536[(alu1+67)];
  var val221 = data2_65536[(alu1+68)];
  var val222 = data2_65536[(alu1+69)];
  var val223 = data2_65536[(alu1+7)];
  var val224 = data2_65536[(alu1+70)];
  var val225 = data2_65536[(alu1+71)];
  var val226 = data2_65536[(alu1+72)];
  var val227 = data2_65536[(alu1+73)];
  var val228 = data2_65536[(alu1+74)];
  var val229 = data2_65536[(alu1+75)];
  var val230 = data2_65536[(alu1+76)];
  var val231 = data2_65536[(alu1+77)];
  var val232 = data2_65536[(alu1+78)];
  var val233 = data2_65536[(alu1+79)];
  var val234 = data2_65536[(alu1+8)];
  var val235 = data2_65536[(alu1+80)];
  var val236 = data2_65536[(alu1+81)];
  var val237 = data2_65536[(alu1+82)];
  var val238 = data2_65536[(alu1+83)];
  var val239 = data2_65536[(alu1+84)];
  var val240 = data2_65536[(alu1+85)];
  var val241 = data2_65536[(alu1+86)];
  var val242 = data2_65536[(alu1+87)];
  var val243 = data2_65536[(alu1+88)];
  var val244 = data2_65536[(alu1+89)];
  var val245 = data2_65536[(alu1+9)];
  var val246 = data2_65536[(alu1+90)];
  var val247 = data2_65536[(alu1+91)];
  var val248 = data2_65536[(alu1+92)];
  var val249 = data2_65536[(alu1+93)];
  var val250 = data2_65536[(alu1+94)];
  var val251 = data2_65536[(alu1+95)];
  var val252 = data2_65536[(alu1+96)];
  var val253 = data2_65536[(alu1+97)];
  var val254 = data2_65536[(alu1+98)];
  var val255 = data2_65536[(alu1+99)];
  var alu2 = (bitcast<i32>((cast4<<6u))+bitcast<i32>((cast5<<2u))+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast1<<16u)));
  data0_524288[alu2] = (((val0*val128)+(val1*val129)+(val40*val168)+(val51*val179)+(val62*val190)+(val73*val201)+(val84*val212)+(val95*val223)+(val106*val234)+(val117*val245)+(val2*val130)+(val13*val141)+(val24*val152)+(val33*val161)+(val34*val162)+(val35*val163)+(val36*val164)+(val37*val165)+(val38*val166)+(val39*val167)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val173)+(val46*val174)+(val47*val175)+(val48*val176)+(val49*val177)+(val50*val178)+(val52*val180)+(val53*val181))*0.1767766922712326f);
  data0_524288[(alu2+1)] = (((val0*val182)+(val1*val183)+(val40*val184)+(val51*val185)+(val62*val186)+(val73*val187)+(val84*val188)+(val95*val189)+(val106*val191)+(val117*val192)+(val2*val193)+(val13*val194)+(val24*val195)+(val33*val196)+(val34*val197)+(val35*val198)+(val36*val199)+(val37*val200)+(val38*val202)+(val39*val203)+(val41*val204)+(val42*val205)+(val43*val206)+(val44*val207)+(val45*val208)+(val46*val209)+(val47*val210)+(val48*val211)+(val49*val213)+(val50*val214)+(val52*val215)+(val53*val216))*0.1767766922712326f);
  data0_524288[(alu2+2)] = (((val0*val217)+(val1*val218)+(val40*val219)+(val51*val220)+(val62*val221)+(val73*val222)+(val84*val224)+(val95*val225)+(val106*val226)+(val117*val227)+(val2*val228)+(val13*val229)+(val24*val230)+(val33*val231)+(val34*val232)+(val35*val233)+(val36*val235)+(val37*val236)+(val38*val237)+(val39*val238)+(val41*val239)+(val42*val240)+(val43*val241)+(val44*val242)+(val45*val243)+(val46*val244)+(val47*val246)+(val48*val247)+(val49*val248)+(val50*val249)+(val52*val250)+(val53*val251))*0.1767766922712326f);
  data0_524288[(alu2+256)] = (((val54*val128)+(val55*val129)+(val56*val168)+(val57*val179)+(val58*val190)+(val59*val201)+(val60*val212)+(val61*val223)+(val63*val234)+(val64*val245)+(val65*val130)+(val66*val141)+(val67*val152)+(val68*val161)+(val69*val162)+(val70*val163)+(val71*val164)+(val72*val165)+(val74*val166)+(val75*val167)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val173)+(val81*val174)+(val82*val175)+(val83*val176)+(val85*val177)+(val86*val178)+(val87*val180)+(val88*val181))*0.1767766922712326f);
  data0_524288[(alu2+257)] = (((val54*val182)+(val55*val183)+(val56*val184)+(val57*val185)+(val58*val186)+(val59*val187)+(val60*val188)+(val61*val189)+(val63*val191)+(val64*val192)+(val65*val193)+(val66*val194)+(val67*val195)+(val68*val196)+(val69*val197)+(val70*val198)+(val71*val199)+(val72*val200)+(val74*val202)+(val75*val203)+(val76*val204)+(val77*val205)+(val78*val206)+(val79*val207)+(val80*val208)+(val81*val209)+(val82*val210)+(val83*val211)+(val85*val213)+(val86*val214)+(val87*val215)+(val88*val216))*0.1767766922712326f);
  data0_524288[(alu2+258)] = (((val54*val217)+(val55*val218)+(val56*val219)+(val57*val220)+(val58*val221)+(val59*val222)+(val60*val224)+(val61*val225)+(val63*val226)+(val64*val227)+(val65*val228)+(val66*val229)+(val67*val230)+(val68*val231)+(val69*val232)+(val70*val233)+(val71*val235)+(val72*val236)+(val74*val237)+(val75*val238)+(val76*val239)+(val77*val240)+(val78*val241)+(val79*val242)+(val80*val243)+(val81*val244)+(val82*val246)+(val83*val247)+(val85*val248)+(val86*val249)+(val87*val250)+(val88*val251))*0.1767766922712326f);
  data0_524288[(alu2+259)] = (((val54*val252)+(val55*val253)+(val56*val254)+(val57*val255)+(val58*val131)+(val59*val132)+(val60*val133)+(val61*val134)+(val63*val135)+(val64*val136)+(val65*val137)+(val66*val138)+(val67*val139)+(val68*val140)+(val69*val142)+(val70*val143)+(val71*val144)+(val72*val145)+(val74*val146)+(val75*val147)+(val76*val148)+(val77*val149)+(val78*val150)+(val79*val151)+(val80*val153)+(val81*val154)+(val82*val155)+(val83*val156)+(val85*val157)+(val86*val158)+(val87*val159)+(val88*val160))*0.1767766922712326f);
  data0_524288[(alu2+3)] = (((val0*val252)+(val1*val253)+(val40*val254)+(val51*val255)+(val62*val131)+(val73*val132)+(val84*val133)+(val95*val134)+(val106*val135)+(val117*val136)+(val2*val137)+(val13*val138)+(val24*val139)+(val33*val140)+(val34*val142)+(val35*val143)+(val36*val144)+(val37*val145)+(val38*val146)+(val39*val147)+(val41*val148)+(val42*val149)+(val43*val150)+(val44*val151)+(val45*val153)+(val46*val154)+(val47*val155)+(val48*val156)+(val49*val157)+(val50*val158)+(val52*val159)+(val53*val160))*0.1767766922712326f);
  data0_524288[(alu2+512)] = (((val89*val128)+(val90*val129)+(val91*val168)+(val92*val179)+(val93*val190)+(val94*val201)+(val96*val212)+(val97*val223)+(val98*val234)+(val99*val245)+(val100*val130)+(val101*val141)+(val102*val152)+(val103*val161)+(val104*val162)+(val105*val163)+(val107*val164)+(val108*val165)+(val109*val166)+(val110*val167)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val173)+(val116*val174)+(val118*val175)+(val119*val176)+(val120*val177)+(val121*val178)+(val122*val180)+(val123*val181))*0.1767766922712326f);
  data0_524288[(alu2+513)] = (((val89*val182)+(val90*val183)+(val91*val184)+(val92*val185)+(val93*val186)+(val94*val187)+(val96*val188)+(val97*val189)+(val98*val191)+(val99*val192)+(val100*val193)+(val101*val194)+(val102*val195)+(val103*val196)+(val104*val197)+(val105*val198)+(val107*val199)+(val108*val200)+(val109*val202)+(val110*val203)+(val111*val204)+(val112*val205)+(val113*val206)+(val114*val207)+(val115*val208)+(val116*val209)+(val118*val210)+(val119*val211)+(val120*val213)+(val121*val214)+(val122*val215)+(val123*val216))*0.1767766922712326f);
  data0_524288[(alu2+514)] = (((val89*val217)+(val90*val218)+(val91*val219)+(val92*val220)+(val93*val221)+(val94*val222)+(val96*val224)+(val97*val225)+(val98*val226)+(val99*val227)+(val100*val228)+(val101*val229)+(val102*val230)+(val103*val231)+(val104*val232)+(val105*val233)+(val107*val235)+(val108*val236)+(val109*val237)+(val110*val238)+(val111*val239)+(val112*val240)+(val113*val241)+(val114*val242)+(val115*val243)+(val116*val244)+(val118*val246)+(val119*val247)+(val120*val248)+(val121*val249)+(val122*val250)+(val123*val251))*0.1767766922712326f);
  data0_524288[(alu2+515)] = (((val89*val252)+(val90*val253)+(val91*val254)+(val92*val255)+(val93*val131)+(val94*val132)+(val96*val133)+(val97*val134)+(val98*val135)+(val99*val136)+(val100*val137)+(val101*val138)+(val102*val139)+(val103*val140)+(val104*val142)+(val105*val143)+(val107*val144)+(val108*val145)+(val109*val146)+(val110*val147)+(val111*val148)+(val112*val149)+(val113*val150)+(val114*val151)+(val115*val153)+(val116*val154)+(val118*val155)+(val119*val156)+(val120*val157)+(val121*val158)+(val122*val159)+(val123*val160))*0.1767766922712326f);
  data0_524288[(alu2+768)] = (((val124*val128)+(val125*val129)+(val126*val168)+(val127*val179)+(val3*val190)+(val4*val201)+(val5*val212)+(val6*val223)+(val7*val234)+(val8*val245)+(val9*val130)+(val10*val141)+(val11*val152)+(val12*val161)+(val14*val162)+(val15*val163)+(val16*val164)+(val17*val165)+(val18*val166)+(val19*val167)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val173)+(val26*val174)+(val27*val175)+(val28*val176)+(val29*val177)+(val30*val178)+(val31*val180)+(val32*val181))*0.1767766922712326f);
  data0_524288[(alu2+769)] = (((val124*val182)+(val125*val183)+(val126*val184)+(val127*val185)+(val3*val186)+(val4*val187)+(val5*val188)+(val6*val189)+(val7*val191)+(val8*val192)+(val9*val193)+(val10*val194)+(val11*val195)+(val12*val196)+(val14*val197)+(val15*val198)+(val16*val199)+(val17*val200)+(val18*val202)+(val19*val203)+(val20*val204)+(val21*val205)+(val22*val206)+(val23*val207)+(val25*val208)+(val26*val209)+(val27*val210)+(val28*val211)+(val29*val213)+(val30*val214)+(val31*val215)+(val32*val216))*0.1767766922712326f);
  data0_524288[(alu2+770)] = (((val124*val217)+(val125*val218)+(val126*val219)+(val127*val220)+(val3*val221)+(val4*val222)+(val5*val224)+(val6*val225)+(val7*val226)+(val8*val227)+(val9*val228)+(val10*val229)+(val11*val230)+(val12*val231)+(val14*val232)+(val15*val233)+(val16*val235)+(val17*val236)+(val18*val237)+(val19*val238)+(val20*val239)+(val21*val240)+(val22*val241)+(val23*val242)+(val25*val243)+(val26*val244)+(val27*val246)+(val28*val247)+(val29*val248)+(val30*val249)+(val31*val250)+(val32*val251))*0.1767766922712326f);
  data0_524288[(alu2+771)] = (((val124*val252)+(val125*val253)+(val126*val254)+(val127*val255)+(val3*val131)+(val4*val132)+(val5*val133)+(val6*val134)+(val7*val135)+(val8*val136)+(val9*val137)+(val10*val138)+(val11*val139)+(val12*val140)+(val14*val142)+(val15*val143)+(val16*val144)+(val17*val145)+(val18*val146)+(val19*val147)+(val20*val148)+(val21*val149)+(val22*val150)+(val23*val151)+(val25*val153)+(val26*val154)+(val27*val155)+(val28*val156)+(val29*val157)+(val30*val158)+(val31*val159)+(val32*val160))*0.1767766922712326f);
}`;

const r_2048_16_16_dispatch_85 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = (f32(-INFINITY));
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val0 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    var val1 = buf0[0];
    var alu1 = select(val1,val0,(val1<val0));
    buf0[0] = alu1;
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = (f32(-INFINITY));
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf1[0];
    var val3 = buf2[Lidx102];
    var alu7 = select(val2,val3,(val2<val3));
    buf1[0] = alu7;
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_2048_16_16_dispatch_86 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var val0 = data2_2048[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val1 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    buf0[0] = (buf0[0]+exp2(((val1-val0)*1.4426950216293335f)));
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf2[Lidx102];
    buf1[0] = (buf1[0]+val2);
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_8_4_16_8_4_4_64_4_dispatch_87 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2048:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_196608:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u))+bitcast<i32>((cast1<<8u)));
  var val0 = data2_2048[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_2048[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_2048[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_2048[alu3];
  var lidx1 = i32(lindex.y); /* 8 */
  var cast3 = bitcast<i32>((bitcast<u32>(lidx1)<<2u));
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var alu20 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast2<<10u))+bitcast<i32>((bitcast<u32>(Lidx0)<<2u))+bitcast<i32>((cast1<<16u)));
    var val4 = data1_524288[alu20];
    var val5 = data1_524288[(alu20+1)];
    var val6 = data1_524288[(alu20+2)];
    var val7 = data1_524288[(alu20+256)];
    var val8 = data1_524288[(alu20+257)];
    var val9 = data1_524288[(alu20+258)];
    var val10 = data1_524288[(alu20+259)];
    var val11 = data1_524288[(alu20+3)];
    var val12 = data1_524288[(alu20+512)];
    var val13 = data1_524288[(alu20+513)];
    var val14 = data1_524288[(alu20+514)];
    var val15 = data1_524288[(alu20+515)];
    var val16 = data1_524288[(alu20+768)];
    var val17 = data1_524288[(alu20+769)];
    var val18 = data1_524288[(alu20+770)];
    var val19 = data1_524288[(alu20+771)];
    var alu21 = (bitcast<i32>((cast1<<5u))+cast3+(Lidx0*3072));
    var val20 = data4_196608[(alu21+1280)];
    var val21 = data4_196608[(alu21+1281)];
    var val22 = data4_196608[(alu21+1282)];
    var val23 = data4_196608[(alu21+1283)];
    var val24 = data4_196608[(alu21+2048)];
    var val25 = data4_196608[(alu21+2049)];
    var val26 = data4_196608[(alu21+2050)];
    var val27 = data4_196608[(alu21+2051)];
    var val28 = data4_196608[(alu21+2816)];
    var val29 = data4_196608[(alu21+2817)];
    var val30 = data4_196608[(alu21+2818)];
    var val31 = data4_196608[(alu21+2819)];
    var val32 = data4_196608[(alu21+512)];
    var val33 = data4_196608[(alu21+513)];
    var val34 = data4_196608[(alu21+514)];
    var val35 = data4_196608[(alu21+515)];
    var alu22 = exp2(((val4-val0)*1.4426950216293335f));
    var alu23 = exp2(((val5-val0)*1.4426950216293335f));
    var alu24 = exp2(((val6-val0)*1.4426950216293335f));
    var alu25 = exp2(((val7-val1)*1.4426950216293335f));
    var alu26 = exp2(((val8-val1)*1.4426950216293335f));
    var alu27 = exp2(((val9-val1)*1.4426950216293335f));
    var alu28 = exp2(((val10-val1)*1.4426950216293335f));
    var alu29 = exp2(((val11-val0)*1.4426950216293335f));
    var alu30 = exp2(((val12-val2)*1.4426950216293335f));
    var alu31 = exp2(((val13-val2)*1.4426950216293335f));
    var alu32 = exp2(((val14-val2)*1.4426950216293335f));
    var alu33 = exp2(((val15-val2)*1.4426950216293335f));
    var alu34 = exp2(((val16-val3)*1.4426950216293335f));
    var alu35 = exp2(((val17-val3)*1.4426950216293335f));
    var alu36 = exp2(((val18-val3)*1.4426950216293335f));
    var alu37 = exp2(((val19-val3)*1.4426950216293335f));
    buf0[0] = (buf0[0]+(alu22*val32)+(alu23*val20)+(alu24*val24)+(alu29*val28));
    buf0[1] = (buf0[1]+(alu22*val33)+(alu23*val21)+(alu24*val25)+(alu29*val29));
    buf0[10] = (buf0[10]+(alu30*val34)+(alu31*val22)+(alu32*val26)+(alu33*val30));
    buf0[11] = (buf0[11]+(alu30*val35)+(alu31*val23)+(alu32*val27)+(alu33*val31));
    buf0[12] = (buf0[12]+(alu34*val32)+(alu35*val20)+(alu36*val24)+(alu37*val28));
    buf0[13] = (buf0[13]+(alu34*val33)+(alu35*val21)+(alu36*val25)+(alu37*val29));
    buf0[14] = (buf0[14]+(alu34*val34)+(alu35*val22)+(alu36*val26)+(alu37*val30));
    buf0[15] = (buf0[15]+(alu34*val35)+(alu35*val23)+(alu36*val27)+(alu37*val31));
    buf0[2] = (buf0[2]+(alu22*val34)+(alu23*val22)+(alu24*val26)+(alu29*val30));
    buf0[3] = (buf0[3]+(alu22*val35)+(alu23*val23)+(alu24*val27)+(alu29*val31));
    buf0[4] = (buf0[4]+(alu25*val32)+(alu26*val20)+(alu27*val24)+(alu28*val28));
    buf0[5] = (buf0[5]+(alu25*val33)+(alu26*val21)+(alu27*val25)+(alu28*val29));
    buf0[6] = (buf0[6]+(alu25*val34)+(alu26*val22)+(alu27*val26)+(alu28*val30));
    buf0[7] = (buf0[7]+(alu25*val35)+(alu26*val23)+(alu27*val27)+(alu28*val31));
    buf0[8] = (buf0[8]+(alu30*val32)+(alu31*val20)+(alu32*val24)+(alu33*val28));
    buf0[9] = (buf0[9]+(alu30*val33)+(alu31*val21)+(alu32*val25)+(alu33*val29));
  }
  var val36 = data3_2048[alu0];
  var val37 = data3_2048[alu1];
  var val38 = data3_2048[alu2];
  var val39 = data3_2048[alu3];
  var alu55 = (bitcast<i32>((cast0<<11u))+bitcast<i32>((cast2<<7u))+cast3+bitcast<i32>((cast1<<13u)));
  var alu56 = (1/val36);
  var alu57 = (1/val37);
  var alu58 = (1/val38);
  var alu59 = (1/val39);
  data0_65536[alu55] = (buf0[0]*alu56);
  data0_65536[(alu55+1)] = (buf0[1]*alu56);
  data0_65536[(alu55+2)] = (buf0[2]*alu56);
  data0_65536[(alu55+3)] = (buf0[3]*alu56);
  data0_65536[(alu55+32)] = (buf0[4]*alu57);
  data0_65536[(alu55+33)] = (buf0[5]*alu57);
  data0_65536[(alu55+34)] = (buf0[6]*alu57);
  data0_65536[(alu55+35)] = (buf0[7]*alu57);
  data0_65536[(alu55+64)] = (buf0[8]*alu58);
  data0_65536[(alu55+65)] = (buf0[9]*alu58);
  data0_65536[(alu55+66)] = (buf0[10]*alu58);
  data0_65536[(alu55+67)] = (buf0[11]*alu58);
  data0_65536[(alu55+96)] = (buf0[12]*alu59);
  data0_65536[(alu55+97)] = (buf0[13]*alu59);
  data0_65536[(alu55+98)] = (buf0[14]*alu59);
  data0_65536[(alu55+99)] = (buf0[15]*alu59);
}`;

const r_8_4_8_16_4_4_8_32_dispatch_88 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_65536:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0_0 = 0; Lidx0_0 < 8; Lidx0_0++) {
    var cast4 = bitcast<u32>(Lidx0_0);
    var alu16 = (bitcast<i32>((cast1<<10u))+bitcast<i32>((cast2<<7u))+bitcast<i32>((cast4<<13u)));
    var val0 = data3_65536[alu16];
    var val1 = data3_65536[(alu16+1)];
    var val2 = data3_65536[(alu16+10)];
    var val3 = data3_65536[(alu16+100)];
    var val4 = data3_65536[(alu16+101)];
    var val5 = data3_65536[(alu16+102)];
    var val6 = data3_65536[(alu16+103)];
    var val7 = data3_65536[(alu16+104)];
    var val8 = data3_65536[(alu16+105)];
    var val9 = data3_65536[(alu16+106)];
    var val10 = data3_65536[(alu16+107)];
    var val11 = data3_65536[(alu16+108)];
    var val12 = data3_65536[(alu16+109)];
    var val13 = data3_65536[(alu16+11)];
    var val14 = data3_65536[(alu16+110)];
    var val15 = data3_65536[(alu16+111)];
    var val16 = data3_65536[(alu16+112)];
    var val17 = data3_65536[(alu16+113)];
    var val18 = data3_65536[(alu16+114)];
    var val19 = data3_65536[(alu16+115)];
    var val20 = data3_65536[(alu16+116)];
    var val21 = data3_65536[(alu16+117)];
    var val22 = data3_65536[(alu16+118)];
    var val23 = data3_65536[(alu16+119)];
    var val24 = data3_65536[(alu16+12)];
    var val25 = data3_65536[(alu16+120)];
    var val26 = data3_65536[(alu16+121)];
    var val27 = data3_65536[(alu16+122)];
    var val28 = data3_65536[(alu16+123)];
    var val29 = data3_65536[(alu16+124)];
    var val30 = data3_65536[(alu16+125)];
    var val31 = data3_65536[(alu16+126)];
    var val32 = data3_65536[(alu16+127)];
    var val33 = data3_65536[(alu16+13)];
    var val34 = data3_65536[(alu16+14)];
    var val35 = data3_65536[(alu16+15)];
    var val36 = data3_65536[(alu16+16)];
    var val37 = data3_65536[(alu16+17)];
    var val38 = data3_65536[(alu16+18)];
    var val39 = data3_65536[(alu16+19)];
    var val40 = data3_65536[(alu16+2)];
    var val41 = data3_65536[(alu16+20)];
    var val42 = data3_65536[(alu16+21)];
    var val43 = data3_65536[(alu16+22)];
    var val44 = data3_65536[(alu16+23)];
    var val45 = data3_65536[(alu16+24)];
    var val46 = data3_65536[(alu16+25)];
    var val47 = data3_65536[(alu16+26)];
    var val48 = data3_65536[(alu16+27)];
    var val49 = data3_65536[(alu16+28)];
    var val50 = data3_65536[(alu16+29)];
    var val51 = data3_65536[(alu16+3)];
    var val52 = data3_65536[(alu16+30)];
    var val53 = data3_65536[(alu16+31)];
    var val54 = data3_65536[(alu16+32)];
    var val55 = data3_65536[(alu16+33)];
    var val56 = data3_65536[(alu16+34)];
    var val57 = data3_65536[(alu16+35)];
    var val58 = data3_65536[(alu16+36)];
    var val59 = data3_65536[(alu16+37)];
    var val60 = data3_65536[(alu16+38)];
    var val61 = data3_65536[(alu16+39)];
    var val62 = data3_65536[(alu16+4)];
    var val63 = data3_65536[(alu16+40)];
    var val64 = data3_65536[(alu16+41)];
    var val65 = data3_65536[(alu16+42)];
    var val66 = data3_65536[(alu16+43)];
    var val67 = data3_65536[(alu16+44)];
    var val68 = data3_65536[(alu16+45)];
    var val69 = data3_65536[(alu16+46)];
    var val70 = data3_65536[(alu16+47)];
    var val71 = data3_65536[(alu16+48)];
    var val72 = data3_65536[(alu16+49)];
    var val73 = data3_65536[(alu16+5)];
    var val74 = data3_65536[(alu16+50)];
    var val75 = data3_65536[(alu16+51)];
    var val76 = data3_65536[(alu16+52)];
    var val77 = data3_65536[(alu16+53)];
    var val78 = data3_65536[(alu16+54)];
    var val79 = data3_65536[(alu16+55)];
    var val80 = data3_65536[(alu16+56)];
    var val81 = data3_65536[(alu16+57)];
    var val82 = data3_65536[(alu16+58)];
    var val83 = data3_65536[(alu16+59)];
    var val84 = data3_65536[(alu16+6)];
    var val85 = data3_65536[(alu16+60)];
    var val86 = data3_65536[(alu16+61)];
    var val87 = data3_65536[(alu16+62)];
    var val88 = data3_65536[(alu16+63)];
    var val89 = data3_65536[(alu16+64)];
    var val90 = data3_65536[(alu16+65)];
    var val91 = data3_65536[(alu16+66)];
    var val92 = data3_65536[(alu16+67)];
    var val93 = data3_65536[(alu16+68)];
    var val94 = data3_65536[(alu16+69)];
    var val95 = data3_65536[(alu16+7)];
    var val96 = data3_65536[(alu16+70)];
    var val97 = data3_65536[(alu16+71)];
    var val98 = data3_65536[(alu16+72)];
    var val99 = data3_65536[(alu16+73)];
    var val100 = data3_65536[(alu16+74)];
    var val101 = data3_65536[(alu16+75)];
    var val102 = data3_65536[(alu16+76)];
    var val103 = data3_65536[(alu16+77)];
    var val104 = data3_65536[(alu16+78)];
    var val105 = data3_65536[(alu16+79)];
    var val106 = data3_65536[(alu16+8)];
    var val107 = data3_65536[(alu16+80)];
    var val108 = data3_65536[(alu16+81)];
    var val109 = data3_65536[(alu16+82)];
    var val110 = data3_65536[(alu16+83)];
    var val111 = data3_65536[(alu16+84)];
    var val112 = data3_65536[(alu16+85)];
    var val113 = data3_65536[(alu16+86)];
    var val114 = data3_65536[(alu16+87)];
    var val115 = data3_65536[(alu16+88)];
    var val116 = data3_65536[(alu16+89)];
    var val117 = data3_65536[(alu16+9)];
    var val118 = data3_65536[(alu16+90)];
    var val119 = data3_65536[(alu16+91)];
    var val120 = data3_65536[(alu16+92)];
    var val121 = data3_65536[(alu16+93)];
    var val122 = data3_65536[(alu16+94)];
    var val123 = data3_65536[(alu16+95)];
    var val124 = data3_65536[(alu16+96)];
    var val125 = data3_65536[(alu16+97)];
    var val126 = data3_65536[(alu16+98)];
    var val127 = data3_65536[(alu16+99)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast4<<5u)));
    var val128 = data4_65536[alu17];
    var val129 = data4_65536[(alu17+1)];
    var val130 = data4_65536[(alu17+10)];
    var val131 = data4_65536[(alu17+11)];
    var val132 = data4_65536[(alu17+12)];
    var val133 = data4_65536[(alu17+13)];
    var val134 = data4_65536[(alu17+14)];
    var val135 = data4_65536[(alu17+15)];
    var val136 = data4_65536[(alu17+16)];
    var val137 = data4_65536[(alu17+17)];
    var val138 = data4_65536[(alu17+18)];
    var val139 = data4_65536[(alu17+19)];
    var val140 = data4_65536[(alu17+2)];
    var val141 = data4_65536[(alu17+20)];
    var val142 = data4_65536[(alu17+21)];
    var val143 = data4_65536[(alu17+22)];
    var val144 = data4_65536[(alu17+23)];
    var val145 = data4_65536[(alu17+24)];
    var val146 = data4_65536[(alu17+25)];
    var val147 = data4_65536[(alu17+256)];
    var val148 = data4_65536[(alu17+257)];
    var val149 = data4_65536[(alu17+258)];
    var val150 = data4_65536[(alu17+259)];
    var val151 = data4_65536[(alu17+26)];
    var val152 = data4_65536[(alu17+260)];
    var val153 = data4_65536[(alu17+261)];
    var val154 = data4_65536[(alu17+262)];
    var val155 = data4_65536[(alu17+263)];
    var val156 = data4_65536[(alu17+264)];
    var val157 = data4_65536[(alu17+265)];
    var val158 = data4_65536[(alu17+266)];
    var val159 = data4_65536[(alu17+267)];
    var val160 = data4_65536[(alu17+268)];
    var val161 = data4_65536[(alu17+269)];
    var val162 = data4_65536[(alu17+27)];
    var val163 = data4_65536[(alu17+270)];
    var val164 = data4_65536[(alu17+271)];
    var val165 = data4_65536[(alu17+272)];
    var val166 = data4_65536[(alu17+273)];
    var val167 = data4_65536[(alu17+274)];
    var val168 = data4_65536[(alu17+275)];
    var val169 = data4_65536[(alu17+276)];
    var val170 = data4_65536[(alu17+277)];
    var val171 = data4_65536[(alu17+278)];
    var val172 = data4_65536[(alu17+279)];
    var val173 = data4_65536[(alu17+28)];
    var val174 = data4_65536[(alu17+280)];
    var val175 = data4_65536[(alu17+281)];
    var val176 = data4_65536[(alu17+282)];
    var val177 = data4_65536[(alu17+283)];
    var val178 = data4_65536[(alu17+284)];
    var val179 = data4_65536[(alu17+285)];
    var val180 = data4_65536[(alu17+286)];
    var val181 = data4_65536[(alu17+287)];
    var val182 = data4_65536[(alu17+29)];
    var val183 = data4_65536[(alu17+3)];
    var val184 = data4_65536[(alu17+30)];
    var val185 = data4_65536[(alu17+31)];
    var val186 = data4_65536[(alu17+4)];
    var val187 = data4_65536[(alu17+5)];
    var val188 = data4_65536[(alu17+512)];
    var val189 = data4_65536[(alu17+513)];
    var val190 = data4_65536[(alu17+514)];
    var val191 = data4_65536[(alu17+515)];
    var val192 = data4_65536[(alu17+516)];
    var val193 = data4_65536[(alu17+517)];
    var val194 = data4_65536[(alu17+518)];
    var val195 = data4_65536[(alu17+519)];
    var val196 = data4_65536[(alu17+520)];
    var val197 = data4_65536[(alu17+521)];
    var val198 = data4_65536[(alu17+522)];
    var val199 = data4_65536[(alu17+523)];
    var val200 = data4_65536[(alu17+524)];
    var val201 = data4_65536[(alu17+525)];
    var val202 = data4_65536[(alu17+526)];
    var val203 = data4_65536[(alu17+527)];
    var val204 = data4_65536[(alu17+528)];
    var val205 = data4_65536[(alu17+529)];
    var val206 = data4_65536[(alu17+530)];
    var val207 = data4_65536[(alu17+531)];
    var val208 = data4_65536[(alu17+532)];
    var val209 = data4_65536[(alu17+533)];
    var val210 = data4_65536[(alu17+534)];
    var val211 = data4_65536[(alu17+535)];
    var val212 = data4_65536[(alu17+536)];
    var val213 = data4_65536[(alu17+537)];
    var val214 = data4_65536[(alu17+538)];
    var val215 = data4_65536[(alu17+539)];
    var val216 = data4_65536[(alu17+540)];
    var val217 = data4_65536[(alu17+541)];
    var val218 = data4_65536[(alu17+542)];
    var val219 = data4_65536[(alu17+543)];
    var val220 = data4_65536[(alu17+6)];
    var val221 = data4_65536[(alu17+7)];
    var val222 = data4_65536[(alu17+768)];
    var val223 = data4_65536[(alu17+769)];
    var val224 = data4_65536[(alu17+770)];
    var val225 = data4_65536[(alu17+771)];
    var val226 = data4_65536[(alu17+772)];
    var val227 = data4_65536[(alu17+773)];
    var val228 = data4_65536[(alu17+774)];
    var val229 = data4_65536[(alu17+775)];
    var val230 = data4_65536[(alu17+776)];
    var val231 = data4_65536[(alu17+777)];
    var val232 = data4_65536[(alu17+778)];
    var val233 = data4_65536[(alu17+779)];
    var val234 = data4_65536[(alu17+780)];
    var val235 = data4_65536[(alu17+781)];
    var val236 = data4_65536[(alu17+782)];
    var val237 = data4_65536[(alu17+783)];
    var val238 = data4_65536[(alu17+784)];
    var val239 = data4_65536[(alu17+785)];
    var val240 = data4_65536[(alu17+786)];
    var val241 = data4_65536[(alu17+787)];
    var val242 = data4_65536[(alu17+788)];
    var val243 = data4_65536[(alu17+789)];
    var val244 = data4_65536[(alu17+790)];
    var val245 = data4_65536[(alu17+791)];
    var val246 = data4_65536[(alu17+792)];
    var val247 = data4_65536[(alu17+793)];
    var val248 = data4_65536[(alu17+794)];
    var val249 = data4_65536[(alu17+795)];
    var val250 = data4_65536[(alu17+796)];
    var val251 = data4_65536[(alu17+797)];
    var val252 = data4_65536[(alu17+798)];
    var val253 = data4_65536[(alu17+799)];
    var val254 = data4_65536[(alu17+8)];
    var val255 = data4_65536[(alu17+9)];
    buf0[0] = (buf0[0]+(val0*val128)+(val1*val129)+(val40*val140)+(val51*val183)+(val62*val186)+(val73*val187)+(val84*val220)+(val95*val221)+(val106*val254)+(val117*val255)+(val2*val130)+(val13*val131)+(val24*val132)+(val33*val133)+(val34*val134)+(val35*val135)+(val36*val136)+(val37*val137)+(val38*val138)+(val39*val139)+(val41*val141)+(val42*val142)+(val43*val143)+(val44*val144)+(val45*val145)+(val46*val146)+(val47*val151)+(val48*val162)+(val49*val173)+(val50*val182)+(val52*val184)+(val53*val185));
    buf0[1] = (buf0[1]+(val0*val147)+(val1*val148)+(val40*val149)+(val51*val150)+(val62*val152)+(val73*val153)+(val84*val154)+(val95*val155)+(val106*val156)+(val117*val157)+(val2*val158)+(val13*val159)+(val24*val160)+(val33*val161)+(val34*val163)+(val35*val164)+(val36*val165)+(val37*val166)+(val38*val167)+(val39*val168)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val174)+(val46*val175)+(val47*val176)+(val48*val177)+(val49*val178)+(val50*val179)+(val52*val180)+(val53*val181));
    buf0[10] = (buf0[10]+(val89*val188)+(val90*val189)+(val91*val190)+(val92*val191)+(val93*val192)+(val94*val193)+(val96*val194)+(val97*val195)+(val98*val196)+(val99*val197)+(val100*val198)+(val101*val199)+(val102*val200)+(val103*val201)+(val104*val202)+(val105*val203)+(val107*val204)+(val108*val205)+(val109*val206)+(val110*val207)+(val111*val208)+(val112*val209)+(val113*val210)+(val114*val211)+(val115*val212)+(val116*val213)+(val118*val214)+(val119*val215)+(val120*val216)+(val121*val217)+(val122*val218)+(val123*val219));
    buf0[11] = (buf0[11]+(val89*val222)+(val90*val223)+(val91*val224)+(val92*val225)+(val93*val226)+(val94*val227)+(val96*val228)+(val97*val229)+(val98*val230)+(val99*val231)+(val100*val232)+(val101*val233)+(val102*val234)+(val103*val235)+(val104*val236)+(val105*val237)+(val107*val238)+(val108*val239)+(val109*val240)+(val110*val241)+(val111*val242)+(val112*val243)+(val113*val244)+(val114*val245)+(val115*val246)+(val116*val247)+(val118*val248)+(val119*val249)+(val120*val250)+(val121*val251)+(val122*val252)+(val123*val253));
    buf0[12] = (buf0[12]+(val124*val128)+(val125*val129)+(val126*val140)+(val127*val183)+(val3*val186)+(val4*val187)+(val5*val220)+(val6*val221)+(val7*val254)+(val8*val255)+(val9*val130)+(val10*val131)+(val11*val132)+(val12*val133)+(val14*val134)+(val15*val135)+(val16*val136)+(val17*val137)+(val18*val138)+(val19*val139)+(val20*val141)+(val21*val142)+(val22*val143)+(val23*val144)+(val25*val145)+(val26*val146)+(val27*val151)+(val28*val162)+(val29*val173)+(val30*val182)+(val31*val184)+(val32*val185));
    buf0[13] = (buf0[13]+(val124*val147)+(val125*val148)+(val126*val149)+(val127*val150)+(val3*val152)+(val4*val153)+(val5*val154)+(val6*val155)+(val7*val156)+(val8*val157)+(val9*val158)+(val10*val159)+(val11*val160)+(val12*val161)+(val14*val163)+(val15*val164)+(val16*val165)+(val17*val166)+(val18*val167)+(val19*val168)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val174)+(val26*val175)+(val27*val176)+(val28*val177)+(val29*val178)+(val30*val179)+(val31*val180)+(val32*val181));
    buf0[14] = (buf0[14]+(val124*val188)+(val125*val189)+(val126*val190)+(val127*val191)+(val3*val192)+(val4*val193)+(val5*val194)+(val6*val195)+(val7*val196)+(val8*val197)+(val9*val198)+(val10*val199)+(val11*val200)+(val12*val201)+(val14*val202)+(val15*val203)+(val16*val204)+(val17*val205)+(val18*val206)+(val19*val207)+(val20*val208)+(val21*val209)+(val22*val210)+(val23*val211)+(val25*val212)+(val26*val213)+(val27*val214)+(val28*val215)+(val29*val216)+(val30*val217)+(val31*val218)+(val32*val219));
    buf0[15] = (buf0[15]+(val124*val222)+(val125*val223)+(val126*val224)+(val127*val225)+(val3*val226)+(val4*val227)+(val5*val228)+(val6*val229)+(val7*val230)+(val8*val231)+(val9*val232)+(val10*val233)+(val11*val234)+(val12*val235)+(val14*val236)+(val15*val237)+(val16*val238)+(val17*val239)+(val18*val240)+(val19*val241)+(val20*val242)+(val21*val243)+(val22*val244)+(val23*val245)+(val25*val246)+(val26*val247)+(val27*val248)+(val28*val249)+(val29*val250)+(val30*val251)+(val31*val252)+(val32*val253));
    buf0[2] = (buf0[2]+(val0*val188)+(val1*val189)+(val40*val190)+(val51*val191)+(val62*val192)+(val73*val193)+(val84*val194)+(val95*val195)+(val106*val196)+(val117*val197)+(val2*val198)+(val13*val199)+(val24*val200)+(val33*val201)+(val34*val202)+(val35*val203)+(val36*val204)+(val37*val205)+(val38*val206)+(val39*val207)+(val41*val208)+(val42*val209)+(val43*val210)+(val44*val211)+(val45*val212)+(val46*val213)+(val47*val214)+(val48*val215)+(val49*val216)+(val50*val217)+(val52*val218)+(val53*val219));
    buf0[3] = (buf0[3]+(val0*val222)+(val1*val223)+(val40*val224)+(val51*val225)+(val62*val226)+(val73*val227)+(val84*val228)+(val95*val229)+(val106*val230)+(val117*val231)+(val2*val232)+(val13*val233)+(val24*val234)+(val33*val235)+(val34*val236)+(val35*val237)+(val36*val238)+(val37*val239)+(val38*val240)+(val39*val241)+(val41*val242)+(val42*val243)+(val43*val244)+(val44*val245)+(val45*val246)+(val46*val247)+(val47*val248)+(val48*val249)+(val49*val250)+(val50*val251)+(val52*val252)+(val53*val253));
    buf0[4] = (buf0[4]+(val54*val128)+(val55*val129)+(val56*val140)+(val57*val183)+(val58*val186)+(val59*val187)+(val60*val220)+(val61*val221)+(val63*val254)+(val64*val255)+(val65*val130)+(val66*val131)+(val67*val132)+(val68*val133)+(val69*val134)+(val70*val135)+(val71*val136)+(val72*val137)+(val74*val138)+(val75*val139)+(val76*val141)+(val77*val142)+(val78*val143)+(val79*val144)+(val80*val145)+(val81*val146)+(val82*val151)+(val83*val162)+(val85*val173)+(val86*val182)+(val87*val184)+(val88*val185));
    buf0[5] = (buf0[5]+(val54*val147)+(val55*val148)+(val56*val149)+(val57*val150)+(val58*val152)+(val59*val153)+(val60*val154)+(val61*val155)+(val63*val156)+(val64*val157)+(val65*val158)+(val66*val159)+(val67*val160)+(val68*val161)+(val69*val163)+(val70*val164)+(val71*val165)+(val72*val166)+(val74*val167)+(val75*val168)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val174)+(val81*val175)+(val82*val176)+(val83*val177)+(val85*val178)+(val86*val179)+(val87*val180)+(val88*val181));
    buf0[6] = (buf0[6]+(val54*val188)+(val55*val189)+(val56*val190)+(val57*val191)+(val58*val192)+(val59*val193)+(val60*val194)+(val61*val195)+(val63*val196)+(val64*val197)+(val65*val198)+(val66*val199)+(val67*val200)+(val68*val201)+(val69*val202)+(val70*val203)+(val71*val204)+(val72*val205)+(val74*val206)+(val75*val207)+(val76*val208)+(val77*val209)+(val78*val210)+(val79*val211)+(val80*val212)+(val81*val213)+(val82*val214)+(val83*val215)+(val85*val216)+(val86*val217)+(val87*val218)+(val88*val219));
    buf0[7] = (buf0[7]+(val54*val222)+(val55*val223)+(val56*val224)+(val57*val225)+(val58*val226)+(val59*val227)+(val60*val228)+(val61*val229)+(val63*val230)+(val64*val231)+(val65*val232)+(val66*val233)+(val67*val234)+(val68*val235)+(val69*val236)+(val70*val237)+(val71*val238)+(val72*val239)+(val74*val240)+(val75*val241)+(val76*val242)+(val77*val243)+(val78*val244)+(val79*val245)+(val80*val246)+(val81*val247)+(val82*val248)+(val83*val249)+(val85*val250)+(val86*val251)+(val87*val252)+(val88*val253));
    buf0[8] = (buf0[8]+(val89*val128)+(val90*val129)+(val91*val140)+(val92*val183)+(val93*val186)+(val94*val187)+(val96*val220)+(val97*val221)+(val98*val254)+(val99*val255)+(val100*val130)+(val101*val131)+(val102*val132)+(val103*val133)+(val104*val134)+(val105*val135)+(val107*val136)+(val108*val137)+(val109*val138)+(val110*val139)+(val111*val141)+(val112*val142)+(val113*val143)+(val114*val144)+(val115*val145)+(val116*val146)+(val118*val151)+(val119*val162)+(val120*val173)+(val121*val182)+(val122*val184)+(val123*val185));
    buf0[9] = (buf0[9]+(val89*val147)+(val90*val148)+(val91*val149)+(val92*val150)+(val93*val152)+(val94*val153)+(val96*val154)+(val97*val155)+(val98*val156)+(val99*val157)+(val100*val158)+(val101*val159)+(val102*val160)+(val103*val161)+(val104*val163)+(val105*val164)+(val107*val165)+(val108*val166)+(val109*val167)+(val110*val168)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val174)+(val116*val175)+(val118*val176)+(val119*val177)+(val120*val178)+(val121*val179)+(val122*val180)+(val123*val181));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<13u))+bitcast<i32>((cast2<<10u)));
  var val256 = data1_65536[alu36];
  var alu37 = (alu36+1);
  var val257 = data1_65536[alu37];
  var alu38 = (alu36+2);
  var val258 = data1_65536[alu38];
  var alu39 = (alu36+256);
  var val259 = data1_65536[alu39];
  var alu40 = (alu36+257);
  var val260 = data1_65536[alu40];
  var alu41 = (alu36+258);
  var val261 = data1_65536[alu41];
  var alu42 = (alu36+259);
  var val262 = data1_65536[alu42];
  var alu43 = (alu36+3);
  var val263 = data1_65536[alu43];
  var alu44 = (alu36+512);
  var val264 = data1_65536[alu44];
  var alu45 = (alu36+513);
  var val265 = data1_65536[alu45];
  var alu46 = (alu36+514);
  var val266 = data1_65536[alu46];
  var alu47 = (alu36+515);
  var val267 = data1_65536[alu47];
  var alu48 = (alu36+768);
  var val268 = data1_65536[alu48];
  var alu49 = (alu36+769);
  var val269 = data1_65536[alu49];
  var alu50 = (alu36+770);
  var val270 = data1_65536[alu50];
  var alu51 = (alu36+771);
  var val271 = data1_65536[alu51];
  var val272 = data2_1536[(alu35+512)];
  var val273 = data2_1536[(alu35+513)];
  var val274 = data2_1536[(alu35+514)];
  var val275 = data2_1536[(alu35+515)];
  var val276 = data5_256[alu35];
  var val277 = data5_256[(alu35+1)];
  var val278 = data5_256[(alu35+2)];
  var val279 = data5_256[(alu35+3)];
  data0_65536[alu36] = (val256+(val272*(buf0[0]+val276)));
  data0_65536[alu37] = (val257+(val273*(buf0[1]+val277)));
  data0_65536[alu38] = (val258+(val274*(buf0[2]+val278)));
  data0_65536[alu39] = (val259+(val272*(buf0[4]+val276)));
  data0_65536[alu40] = (val260+(val273*(buf0[5]+val277)));
  data0_65536[alu41] = (val261+(val274*(buf0[6]+val278)));
  data0_65536[alu42] = (val262+(val275*(buf0[7]+val279)));
  data0_65536[alu43] = (val263+(val275*(buf0[3]+val279)));
  data0_65536[alu44] = (val264+(val272*(buf0[8]+val276)));
  data0_65536[alu45] = (val265+(val273*(buf0[9]+val277)));
  data0_65536[alu46] = (val266+(val274*(buf0[10]+val278)));
  data0_65536[alu47] = (val267+(val275*(buf0[11]+val279)));
  data0_65536[alu48] = (val268+(val272*(buf0[12]+val276)));
  data0_65536[alu49] = (val269+(val273*(buf0[13]+val277)));
  data0_65536[alu50] = (val270+(val274*(buf0[14]+val278)));
  data0_65536[alu51] = (val271+(val275*(buf0[15]+val279)));
}`;

const r_16_8_4_4_32_dispatch_89 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_90 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var val21 = data3_256[(alu0+1)];
  var val22 = data3_256[(alu0+2)];
  var val23 = data3_256[(alu0+3)];
  var val24 = data4_1536[(alu0+1024)];
  var val25 = data4_1536[(alu0+1025)];
  var val26 = data4_1536[(alu0+1026)];
  var val27 = data4_1536[(alu0+1027)];
  var val28 = data4_1536[(alu0+768)];
  var val29 = data4_1536[(alu0+769)];
  var val30 = data4_1536[(alu0+770)];
  var val31 = data4_1536[(alu0+771)];
  var alu18 = (1/val16);
  var alu19 = (1/val17);
  var alu20 = (1/val18);
  var alu21 = (1/val19);
  var alu22 = (1.0f+val24);
  var alu23 = (1.0f+val25);
  var alu24 = (1.0f+val26);
  var alu25 = (1.0f+val27);
  data0_65536[alu1] = ((val0*alu18*val20*alu22)+val28);
  data0_65536[alu2] = ((val1*alu18*val21*alu23)+val29);
  data0_65536[alu3] = ((val2*alu18*val22*alu24)+val30);
  data0_65536[alu4] = ((val3*alu19*val20*alu22)+val28);
  data0_65536[alu5] = ((val4*alu19*val21*alu23)+val29);
  data0_65536[alu6] = ((val5*alu19*val22*alu24)+val30);
  data0_65536[alu7] = ((val6*alu19*val23*alu25)+val31);
  data0_65536[alu8] = ((val7*alu18*val23*alu25)+val31);
  data0_65536[alu9] = ((val8*alu20*val20*alu22)+val28);
  data0_65536[alu10] = ((val9*alu20*val21*alu23)+val29);
  data0_65536[alu11] = ((val10*alu20*val22*alu24)+val30);
  data0_65536[alu12] = ((val11*alu20*val23*alu25)+val31);
  data0_65536[alu13] = ((val12*alu21*val20*alu22)+val28);
  data0_65536[alu14] = ((val13*alu21*val21*alu23)+val29);
  data0_65536[alu15] = ((val14*alu21*val22*alu24)+val30);
  data0_65536[alu16] = ((val15*alu21*val23*alu25)+val31);
}`;

const r_8_22_8_16_4_4_64_4_dispatch_91 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_360448:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_360448:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1408:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 22 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast2 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast2);
    var val0 = data1_65536[alu16];
    var val1 = data1_65536[(alu16+1)];
    var val2 = data1_65536[(alu16+2)];
    var val3 = data1_65536[(alu16+256)];
    var val4 = data1_65536[(alu16+257)];
    var val5 = data1_65536[(alu16+258)];
    var val6 = data1_65536[(alu16+259)];
    var val7 = data1_65536[(alu16+3)];
    var val8 = data1_65536[(alu16+512)];
    var val9 = data1_65536[(alu16+513)];
    var val10 = data1_65536[(alu16+514)];
    var val11 = data1_65536[(alu16+515)];
    var val12 = data1_65536[(alu16+768)];
    var val13 = data1_65536[(alu16+769)];
    var val14 = data1_65536[(alu16+770)];
    var val15 = data1_65536[(alu16+771)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast1<<10u))+cast2);
    var val16 = data2_360448[alu17];
    var val17 = data2_360448[(alu17+1)];
    var val18 = data2_360448[(alu17+2)];
    var val19 = data2_360448[(alu17+256)];
    var val20 = data2_360448[(alu17+257)];
    var val21 = data2_360448[(alu17+258)];
    var val22 = data2_360448[(alu17+259)];
    var val23 = data2_360448[(alu17+3)];
    var val24 = data2_360448[(alu17+512)];
    var val25 = data2_360448[(alu17+513)];
    var val26 = data2_360448[(alu17+514)];
    var val27 = data2_360448[(alu17+515)];
    var val28 = data2_360448[(alu17+768)];
    var val29 = data2_360448[(alu17+769)];
    var val30 = data2_360448[(alu17+770)];
    var val31 = data2_360448[(alu17+771)];
    buf0[0] = (buf0[0]+(val0*val16)+(val1*val17)+(val2*val18)+(val7*val23));
    buf0[1] = (buf0[1]+(val0*val19)+(val1*val20)+(val2*val21)+(val7*val22));
    buf0[10] = (buf0[10]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    buf0[11] = (buf0[11]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    buf0[12] = (buf0[12]+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val23));
    buf0[13] = (buf0[13]+(val12*val19)+(val13*val20)+(val14*val21)+(val15*val22));
    buf0[14] = (buf0[14]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    buf0[15] = (buf0[15]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    buf0[2] = (buf0[2]+(val0*val24)+(val1*val25)+(val2*val26)+(val7*val27));
    buf0[3] = (buf0[3]+(val0*val28)+(val1*val29)+(val2*val30)+(val7*val31));
    buf0[4] = (buf0[4]+(val3*val16)+(val4*val17)+(val5*val18)+(val6*val23));
    buf0[5] = (buf0[5]+(val3*val19)+(val4*val20)+(val5*val21)+(val6*val22));
    buf0[6] = (buf0[6]+(val3*val24)+(val4*val25)+(val5*val26)+(val6*val27));
    buf0[7] = (buf0[7]+(val3*val28)+(val4*val29)+(val5*val30)+(val6*val31));
    buf0[8] = (buf0[8]+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val23));
    buf0[9] = (buf0[9]+(val8*val19)+(val9*val20)+(val10*val21)+(val11*val22));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)));
  var val32 = data3_1408[alu35];
  var val33 = data3_1408[(alu35+1)];
  var val34 = data3_1408[(alu35+2)];
  var val35 = data3_1408[(alu35+3)];
  var alu36 = (alu35+(gidx1*45056)+(lidx0*5632));
  data0_360448[alu36] = (buf0[0]+val32);
  data0_360448[(alu36+1)] = (buf0[1]+val33);
  data0_360448[(alu36+1408)] = (buf0[4]+val32);
  data0_360448[(alu36+1409)] = (buf0[5]+val33);
  data0_360448[(alu36+1410)] = (buf0[6]+val34);
  data0_360448[(alu36+1411)] = (buf0[7]+val35);
  data0_360448[(alu36+2)] = (buf0[2]+val34);
  data0_360448[(alu36+2816)] = (buf0[8]+val32);
  data0_360448[(alu36+2817)] = (buf0[9]+val33);
  data0_360448[(alu36+2818)] = (buf0[10]+val34);
  data0_360448[(alu36+2819)] = (buf0[11]+val35);
  data0_360448[(alu36+3)] = (buf0[3]+val35);
  data0_360448[(alu36+4224)] = (buf0[12]+val32);
  data0_360448[(alu36+4225)] = (buf0[13]+val33);
  data0_360448[(alu36+4226)] = (buf0[14]+val34);
  data0_360448[(alu36+4227)] = (buf0[15]+val35);
}`;

const r_8_4_8_16_4_4_176_4_dispatch_92 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_360448:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_180224:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 176; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = ((gidx1*45056)+(lidx0*5632)+cast0);
    var val0 = data3_360448[(alu16+1)];
    var val1 = data3_360448[(alu16+1408)];
    var val2 = data3_360448[(alu16+1409)];
    var val3 = data3_360448[(alu16+1410)];
    var val4 = data3_360448[(alu16+1411)];
    var val5 = data3_360448[(alu16+2)];
    var val6 = data3_360448[(alu16+2112)];
    var val7 = data3_360448[(alu16+2113)];
    var val8 = data3_360448[(alu16+2114)];
    var val9 = data3_360448[(alu16+2115)];
    var val10 = data3_360448[(alu16+2816)];
    var val11 = data3_360448[(alu16+2817)];
    var val12 = data3_360448[(alu16+2818)];
    var val13 = data3_360448[(alu16+2819)];
    var val14 = data3_360448[(alu16+3)];
    var val15 = data3_360448[(alu16+3520)];
    var val16 = data3_360448[(alu16+3521)];
    var val17 = data3_360448[(alu16+3522)];
    var val18 = data3_360448[(alu16+3523)];
    var val19 = data3_360448[(alu16+4224)];
    var val20 = data3_360448[(alu16+4225)];
    var val21 = data3_360448[(alu16+4226)];
    var val22 = data3_360448[(alu16+4227)];
    var val23 = data3_360448[(alu16+4928)];
    var val24 = data3_360448[(alu16+4929)];
    var val25 = data3_360448[(alu16+4930)];
    var val26 = data3_360448[(alu16+4931)];
    var val27 = data3_360448[(alu16+704)];
    var val28 = data3_360448[(alu16+705)];
    var val29 = data3_360448[(alu16+706)];
    var val30 = data3_360448[(alu16+707)];
    var val31 = data3_360448[alu16];
    var alu17 = ((gidx0*45056)+(lidx1*2816)+cast0);
    var val32 = data4_180224[(alu17+1)];
    var val33 = data4_180224[(alu17+1408)];
    var val34 = data4_180224[(alu17+1409)];
    var val35 = data4_180224[(alu17+1410)];
    var val36 = data4_180224[(alu17+1411)];
    var val37 = data4_180224[(alu17+2)];
    var val38 = data4_180224[(alu17+2112)];
    var val39 = data4_180224[(alu17+2113)];
    var val40 = data4_180224[(alu17+2114)];
    var val41 = data4_180224[(alu17+2115)];
    var val42 = data4_180224[(alu17+3)];
    var val43 = data4_180224[(alu17+704)];
    var val44 = data4_180224[(alu17+705)];
    var val45 = data4_180224[(alu17+706)];
    var val46 = data4_180224[(alu17+707)];
    var val47 = data4_180224[alu17];
    var alu18 = (val0*(1/(1.0f+exp2((val0*-1.4426950216293335f))))*val28);
    var alu19 = (val31*(1/(1.0f+exp2((val31*-1.4426950216293335f))))*val27);
    var alu20 = (val5*(1/(1.0f+exp2((val5*-1.4426950216293335f))))*val29);
    var alu21 = (val14*(1/(1.0f+exp2((val14*-1.4426950216293335f))))*val30);
    var alu22 = (val10*(1/(1.0f+exp2((val10*-1.4426950216293335f))))*val15);
    var alu23 = (val11*(1/(1.0f+exp2((val11*-1.4426950216293335f))))*val16);
    var alu24 = (val12*(1/(1.0f+exp2((val12*-1.4426950216293335f))))*val17);
    var alu25 = (val13*(1/(1.0f+exp2((val13*-1.4426950216293335f))))*val18);
    var alu26 = (val19*(1/(1.0f+exp2((val19*-1.4426950216293335f))))*val23);
    var alu27 = (val20*(1/(1.0f+exp2((val20*-1.4426950216293335f))))*val24);
    var alu28 = (val21*(1/(1.0f+exp2((val21*-1.4426950216293335f))))*val25);
    var alu29 = (val22*(1/(1.0f+exp2((val22*-1.4426950216293335f))))*val26);
    var alu30 = (val1*(1/(1.0f+exp2((val1*-1.4426950216293335f))))*val6);
    var alu31 = (val2*(1/(1.0f+exp2((val2*-1.4426950216293335f))))*val7);
    var alu32 = (val3*(1/(1.0f+exp2((val3*-1.4426950216293335f))))*val8);
    var alu33 = (val4*(1/(1.0f+exp2((val4*-1.4426950216293335f))))*val9);
    buf0[0] = (buf0[0]+(alu19*val47)+(alu18*val32)+(alu20*val37)+(alu21*val42));
    buf0[1] = (buf0[1]+(alu19*val43)+(alu18*val44)+(alu20*val45)+(alu21*val46));
    buf0[10] = (buf0[10]+(alu22*val33)+(alu23*val34)+(alu24*val35)+(alu25*val36));
    buf0[11] = (buf0[11]+(alu22*val38)+(alu23*val39)+(alu24*val40)+(alu25*val41));
    buf0[12] = (buf0[12]+(alu26*val47)+(alu27*val32)+(alu28*val37)+(alu29*val42));
    buf0[13] = (buf0[13]+(alu26*val43)+(alu27*val44)+(alu28*val45)+(alu29*val46));
    buf0[14] = (buf0[14]+(alu26*val33)+(alu27*val34)+(alu28*val35)+(alu29*val36));
    buf0[15] = (buf0[15]+(alu26*val38)+(alu27*val39)+(alu28*val40)+(alu29*val41));
    buf0[2] = (buf0[2]+(alu19*val33)+(alu18*val34)+(alu20*val35)+(alu21*val36));
    buf0[3] = (buf0[3]+(alu19*val38)+(alu18*val39)+(alu20*val40)+(alu21*val41));
    buf0[4] = (buf0[4]+(alu30*val47)+(alu31*val32)+(alu32*val37)+(alu33*val42));
    buf0[5] = (buf0[5]+(alu30*val43)+(alu31*val44)+(alu32*val45)+(alu33*val46));
    buf0[6] = (buf0[6]+(alu30*val33)+(alu31*val34)+(alu32*val35)+(alu33*val36));
    buf0[7] = (buf0[7]+(alu30*val38)+(alu31*val39)+(alu32*val40)+(alu33*val41));
    buf0[8] = (buf0[8]+(alu22*val47)+(alu23*val32)+(alu24*val37)+(alu25*val42));
    buf0[9] = (buf0[9]+(alu22*val43)+(alu23*val44)+(alu24*val45)+(alu25*val46));
  }
  var alu51 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu52 = (alu51+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u)));
  var val48 = data1_65536[alu52];
  var alu53 = (alu52+1);
  var val49 = data1_65536[alu53];
  var alu54 = (alu52+2);
  var val50 = data1_65536[alu54];
  var alu55 = (alu52+256);
  var val51 = data1_65536[alu55];
  var alu56 = (alu52+257);
  var val52 = data1_65536[alu56];
  var alu57 = (alu52+258);
  var val53 = data1_65536[alu57];
  var alu58 = (alu52+259);
  var val54 = data1_65536[alu58];
  var alu59 = (alu52+3);
  var val55 = data1_65536[alu59];
  var alu60 = (alu52+512);
  var val56 = data1_65536[alu60];
  var alu61 = (alu52+513);
  var val57 = data1_65536[alu61];
  var alu62 = (alu52+514);
  var val58 = data1_65536[alu62];
  var alu63 = (alu52+515);
  var val59 = data1_65536[alu63];
  var alu64 = (alu52+768);
  var val60 = data1_65536[alu64];
  var alu65 = (alu52+769);
  var val61 = data1_65536[alu65];
  var alu66 = (alu52+770);
  var val62 = data1_65536[alu66];
  var alu67 = (alu52+771);
  var val63 = data1_65536[alu67];
  var val64 = data2_1536[(alu51+1280)];
  var val65 = data2_1536[(alu51+1281)];
  var val66 = data2_1536[(alu51+1282)];
  var val67 = data2_1536[(alu51+1283)];
  var val68 = data5_256[alu51];
  var val69 = data5_256[(alu51+1)];
  var val70 = data5_256[(alu51+2)];
  var val71 = data5_256[(alu51+3)];
  data0_65536[alu52] = (val48+(val64*(buf0[0]+val68)));
  data0_65536[alu53] = (val49+(val65*(buf0[1]+val69)));
  data0_65536[alu54] = (val50+(val66*(buf0[2]+val70)));
  data0_65536[alu55] = (val51+(val64*(buf0[4]+val68)));
  data0_65536[alu56] = (val52+(val65*(buf0[5]+val69)));
  data0_65536[alu57] = (val53+(val66*(buf0[6]+val70)));
  data0_65536[alu58] = (val54+(val67*(buf0[7]+val71)));
  data0_65536[alu59] = (val55+(val67*(buf0[3]+val71)));
  data0_65536[alu60] = (val56+(val64*(buf0[8]+val68)));
  data0_65536[alu61] = (val57+(val65*(buf0[9]+val69)));
  data0_65536[alu62] = (val58+(val66*(buf0[10]+val70)));
  data0_65536[alu63] = (val59+(val67*(buf0[11]+val71)));
  data0_65536[alu64] = (val60+(val64*(buf0[12]+val68)));
  data0_65536[alu65] = (val61+(val65*(buf0[13]+val69)));
  data0_65536[alu66] = (val62+(val66*(buf0[14]+val70)));
  data0_65536[alu67] = (val63+(val67*(buf0[15]+val71)));
}`;

const r_16_8_4_4_32_dispatch_93 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_94 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var alu18 = (alu0+1);
  var val21 = data3_256[alu18];
  var alu19 = (alu0+2);
  var val22 = data3_256[alu19];
  var alu20 = (alu0+3);
  var val23 = data3_256[alu20];
  var val24 = data4_1536[alu0];
  var val25 = data4_1536[alu18];
  var val26 = data4_1536[alu19];
  var val27 = data4_1536[(alu0+256)];
  var val28 = data4_1536[(alu0+257)];
  var val29 = data4_1536[(alu0+258)];
  var val30 = data4_1536[(alu0+259)];
  var val31 = data4_1536[alu20];
  var alu21 = (1/val16);
  var alu22 = (1/val17);
  var alu23 = (1/val18);
  var alu24 = (1/val19);
  var alu25 = (1.0f+val27);
  var alu26 = (1.0f+val28);
  var alu27 = (1.0f+val29);
  var alu28 = (1.0f+val30);
  data0_65536[alu1] = ((val0*alu21*val20*alu25)+val24);
  data0_65536[alu2] = ((val1*alu21*val21*alu26)+val25);
  data0_65536[alu3] = ((val2*alu21*val22*alu27)+val26);
  data0_65536[alu4] = ((val3*alu22*val20*alu25)+val24);
  data0_65536[alu5] = ((val4*alu22*val21*alu26)+val25);
  data0_65536[alu6] = ((val5*alu22*val22*alu27)+val26);
  data0_65536[alu7] = ((val6*alu22*val23*alu28)+val31);
  data0_65536[alu8] = ((val7*alu21*val23*alu28)+val31);
  data0_65536[alu9] = ((val8*alu23*val20*alu25)+val24);
  data0_65536[alu10] = ((val9*alu23*val21*alu26)+val25);
  data0_65536[alu11] = ((val10*alu23*val22*alu27)+val26);
  data0_65536[alu12] = ((val11*alu23*val23*alu28)+val31);
  data0_65536[alu13] = ((val12*alu24*val20*alu25)+val24);
  data0_65536[alu14] = ((val13*alu24*val21*alu26)+val25);
  data0_65536[alu15] = ((val14*alu24*val22*alu27)+val26);
  data0_65536[alu16] = ((val15*alu24*val23*alu28)+val31);
}`;

const r_8_16_8_16_3_4_64_4_dispatch_95 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_196608:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu12 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast0);
    var val0 = data1_65536[alu12];
    var val1 = data1_65536[(alu12+1)];
    var val2 = data1_65536[(alu12+2)];
    var val3 = data1_65536[(alu12+256)];
    var val4 = data1_65536[(alu12+257)];
    var val5 = data1_65536[(alu12+258)];
    var val6 = data1_65536[(alu12+259)];
    var val7 = data1_65536[(alu12+3)];
    var val8 = data1_65536[(alu12+512)];
    var val9 = data1_65536[(alu12+513)];
    var val10 = data1_65536[(alu12+514)];
    var val11 = data1_65536[(alu12+515)];
    var val12 = data1_65536[(alu12+768)];
    var val13 = data1_65536[(alu12+769)];
    var val14 = data1_65536[(alu12+770)];
    var val15 = data1_65536[(alu12+771)];
    var alu13 = ((gidx0*12288)+(lidx1*768)+cast0);
    var val16 = data2_196608[(alu13+1)];
    var val17 = data2_196608[(alu13+2)];
    var val18 = data2_196608[(alu13+256)];
    var val19 = data2_196608[(alu13+257)];
    var val20 = data2_196608[(alu13+258)];
    var val21 = data2_196608[(alu13+259)];
    var val22 = data2_196608[(alu13+3)];
    var val23 = data2_196608[(alu13+512)];
    var val24 = data2_196608[(alu13+513)];
    var val25 = data2_196608[(alu13+514)];
    var val26 = data2_196608[(alu13+515)];
    var val27 = data2_196608[alu13];
    buf0[0] = (buf0[0]+(val0*val27)+(val1*val16)+(val2*val17)+(val7*val22));
    buf0[1] = (buf0[1]+(val0*val18)+(val1*val19)+(val2*val20)+(val7*val21));
    buf0[10] = (buf0[10]+(val12*val18)+(val13*val19)+(val14*val20)+(val15*val21));
    buf0[11] = (buf0[11]+(val12*val23)+(val13*val24)+(val14*val25)+(val15*val26));
    buf0[2] = (buf0[2]+(val0*val23)+(val1*val24)+(val2*val25)+(val7*val26));
    buf0[3] = (buf0[3]+(val3*val27)+(val4*val16)+(val5*val17)+(val6*val22));
    buf0[4] = (buf0[4]+(val3*val18)+(val4*val19)+(val5*val20)+(val6*val21));
    buf0[5] = (buf0[5]+(val3*val23)+(val4*val24)+(val5*val25)+(val6*val26));
    buf0[6] = (buf0[6]+(val8*val27)+(val9*val16)+(val10*val17)+(val11*val22));
    buf0[7] = (buf0[7]+(val8*val18)+(val9*val19)+(val10*val20)+(val11*val21));
    buf0[8] = (buf0[8]+(val8*val23)+(val9*val24)+(val10*val25)+(val11*val26));
    buf0[9] = (buf0[9]+(val12*val27)+(val13*val16)+(val14*val17)+(val15*val22));
  }
  var alu27 = ((gidx0*48)+(lidx1*3));
  var val28 = data3_768[(alu27+1)];
  var val29 = data3_768[(alu27+2)];
  var val30 = data3_768[alu27];
  var alu28 = (alu27+(gidx1*24576)+(lidx0*3072));
  data0_196608[(alu28+1)] = (buf0[1]+val28);
  data0_196608[(alu28+1536)] = (buf0[6]+val30);
  data0_196608[(alu28+1537)] = (buf0[7]+val28);
  data0_196608[(alu28+1538)] = (buf0[8]+val29);
  data0_196608[(alu28+2)] = (buf0[2]+val29);
  data0_196608[(alu28+2304)] = (buf0[9]+val30);
  data0_196608[(alu28+2305)] = (buf0[10]+val28);
  data0_196608[(alu28+2306)] = (buf0[11]+val29);
  data0_196608[(alu28+768)] = (buf0[3]+val30);
  data0_196608[(alu28+769)] = (buf0[4]+val28);
  data0_196608[(alu28+770)] = (buf0[5]+val29);
  data0_196608[alu28] = (buf0[0]+val30);
}`;

const r_8_16_8_4_4_4_dispatch_96 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[alu4];
    var val1 = data1_196608[(alu4+3072)];
    var val2 = data1_196608[(alu4+6144)];
    var val3 = data1_196608[(alu4+9216)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const r_8_16_8_4_4_4_dispatch_97 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[(alu4+256)];
    var val1 = data1_196608[(alu4+3328)];
    var val2 = data1_196608[(alu4+6400)];
    var val3 = data1_196608[(alu4+9472)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const E_2_32_8_16_2_4_dispatch_98 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+1)];
  var val1 = data1_196608[(alu0+32)];
  var val2 = data1_196608[(alu0+33)];
  var val3 = data1_196608[(alu0+64)];
  var val4 = data1_196608[(alu0+65)];
  var val5 = data1_196608[(alu0+96)];
  var val6 = data1_196608[(alu0+97)];
  var val7 = data1_196608[alu0];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val13);
  var alu10 = (val7*alu5*val12);
  var alu11 = (val1*alu6*val12);
  var alu12 = (val2*alu6*val13);
  var alu13 = (val3*alu7*val12);
  var alu14 = (val4*alu7*val13);
  var alu15 = (val5*alu8*val12);
  var alu16 = (val6*alu8*val13);
  data0_65536[alu4] = ((alu10*val14)-(alu9*val16));
  data0_65536[(alu4+1)] = ((alu9*val15)+(alu10*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const E_2_32_8_16_2_4_dispatch_99 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+256)];
  var val1 = data1_196608[(alu0+257)];
  var val2 = data1_196608[(alu0+288)];
  var val3 = data1_196608[(alu0+289)];
  var val4 = data1_196608[(alu0+320)];
  var val5 = data1_196608[(alu0+321)];
  var val6 = data1_196608[(alu0+352)];
  var val7 = data1_196608[(alu0+353)];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val12);
  var alu10 = (val1*alu5*val13);
  var alu11 = (val2*alu6*val12);
  var alu12 = (val3*alu6*val13);
  var alu13 = (val4*alu7*val12);
  var alu14 = (val5*alu7*val13);
  var alu15 = (val6*alu8*val12);
  var alu16 = (val7*alu8*val13);
  data0_65536[alu4] = ((alu9*val14)-(alu10*val16));
  data0_65536[(alu4+1)] = ((alu10*val15)+(alu9*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const r_8_8_4_8_16_4_4_32_dispatch_100 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_524288:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
  var cast2 = bitcast<i32>((cast1<<13u));
  var cast3 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<10u))+bitcast<i32>((cast3<<7u))+cast2);
  var val0 = data1_65536[alu0];
  var val1 = data1_65536[(alu0+1)];
  var val2 = data1_65536[(alu0+10)];
  var val3 = data1_65536[(alu0+100)];
  var val4 = data1_65536[(alu0+101)];
  var val5 = data1_65536[(alu0+102)];
  var val6 = data1_65536[(alu0+103)];
  var val7 = data1_65536[(alu0+104)];
  var val8 = data1_65536[(alu0+105)];
  var val9 = data1_65536[(alu0+106)];
  var val10 = data1_65536[(alu0+107)];
  var val11 = data1_65536[(alu0+108)];
  var val12 = data1_65536[(alu0+109)];
  var val13 = data1_65536[(alu0+11)];
  var val14 = data1_65536[(alu0+110)];
  var val15 = data1_65536[(alu0+111)];
  var val16 = data1_65536[(alu0+112)];
  var val17 = data1_65536[(alu0+113)];
  var val18 = data1_65536[(alu0+114)];
  var val19 = data1_65536[(alu0+115)];
  var val20 = data1_65536[(alu0+116)];
  var val21 = data1_65536[(alu0+117)];
  var val22 = data1_65536[(alu0+118)];
  var val23 = data1_65536[(alu0+119)];
  var val24 = data1_65536[(alu0+12)];
  var val25 = data1_65536[(alu0+120)];
  var val26 = data1_65536[(alu0+121)];
  var val27 = data1_65536[(alu0+122)];
  var val28 = data1_65536[(alu0+123)];
  var val29 = data1_65536[(alu0+124)];
  var val30 = data1_65536[(alu0+125)];
  var val31 = data1_65536[(alu0+126)];
  var val32 = data1_65536[(alu0+127)];
  var val33 = data1_65536[(alu0+13)];
  var val34 = data1_65536[(alu0+14)];
  var val35 = data1_65536[(alu0+15)];
  var val36 = data1_65536[(alu0+16)];
  var val37 = data1_65536[(alu0+17)];
  var val38 = data1_65536[(alu0+18)];
  var val39 = data1_65536[(alu0+19)];
  var val40 = data1_65536[(alu0+2)];
  var val41 = data1_65536[(alu0+20)];
  var val42 = data1_65536[(alu0+21)];
  var val43 = data1_65536[(alu0+22)];
  var val44 = data1_65536[(alu0+23)];
  var val45 = data1_65536[(alu0+24)];
  var val46 = data1_65536[(alu0+25)];
  var val47 = data1_65536[(alu0+26)];
  var val48 = data1_65536[(alu0+27)];
  var val49 = data1_65536[(alu0+28)];
  var val50 = data1_65536[(alu0+29)];
  var val51 = data1_65536[(alu0+3)];
  var val52 = data1_65536[(alu0+30)];
  var val53 = data1_65536[(alu0+31)];
  var val54 = data1_65536[(alu0+32)];
  var val55 = data1_65536[(alu0+33)];
  var val56 = data1_65536[(alu0+34)];
  var val57 = data1_65536[(alu0+35)];
  var val58 = data1_65536[(alu0+36)];
  var val59 = data1_65536[(alu0+37)];
  var val60 = data1_65536[(alu0+38)];
  var val61 = data1_65536[(alu0+39)];
  var val62 = data1_65536[(alu0+4)];
  var val63 = data1_65536[(alu0+40)];
  var val64 = data1_65536[(alu0+41)];
  var val65 = data1_65536[(alu0+42)];
  var val66 = data1_65536[(alu0+43)];
  var val67 = data1_65536[(alu0+44)];
  var val68 = data1_65536[(alu0+45)];
  var val69 = data1_65536[(alu0+46)];
  var val70 = data1_65536[(alu0+47)];
  var val71 = data1_65536[(alu0+48)];
  var val72 = data1_65536[(alu0+49)];
  var val73 = data1_65536[(alu0+5)];
  var val74 = data1_65536[(alu0+50)];
  var val75 = data1_65536[(alu0+51)];
  var val76 = data1_65536[(alu0+52)];
  var val77 = data1_65536[(alu0+53)];
  var val78 = data1_65536[(alu0+54)];
  var val79 = data1_65536[(alu0+55)];
  var val80 = data1_65536[(alu0+56)];
  var val81 = data1_65536[(alu0+57)];
  var val82 = data1_65536[(alu0+58)];
  var val83 = data1_65536[(alu0+59)];
  var val84 = data1_65536[(alu0+6)];
  var val85 = data1_65536[(alu0+60)];
  var val86 = data1_65536[(alu0+61)];
  var val87 = data1_65536[(alu0+62)];
  var val88 = data1_65536[(alu0+63)];
  var val89 = data1_65536[(alu0+64)];
  var val90 = data1_65536[(alu0+65)];
  var val91 = data1_65536[(alu0+66)];
  var val92 = data1_65536[(alu0+67)];
  var val93 = data1_65536[(alu0+68)];
  var val94 = data1_65536[(alu0+69)];
  var val95 = data1_65536[(alu0+7)];
  var val96 = data1_65536[(alu0+70)];
  var val97 = data1_65536[(alu0+71)];
  var val98 = data1_65536[(alu0+72)];
  var val99 = data1_65536[(alu0+73)];
  var val100 = data1_65536[(alu0+74)];
  var val101 = data1_65536[(alu0+75)];
  var val102 = data1_65536[(alu0+76)];
  var val103 = data1_65536[(alu0+77)];
  var val104 = data1_65536[(alu0+78)];
  var val105 = data1_65536[(alu0+79)];
  var val106 = data1_65536[(alu0+8)];
  var val107 = data1_65536[(alu0+80)];
  var val108 = data1_65536[(alu0+81)];
  var val109 = data1_65536[(alu0+82)];
  var val110 = data1_65536[(alu0+83)];
  var val111 = data1_65536[(alu0+84)];
  var val112 = data1_65536[(alu0+85)];
  var val113 = data1_65536[(alu0+86)];
  var val114 = data1_65536[(alu0+87)];
  var val115 = data1_65536[(alu0+88)];
  var val116 = data1_65536[(alu0+89)];
  var val117 = data1_65536[(alu0+9)];
  var val118 = data1_65536[(alu0+90)];
  var val119 = data1_65536[(alu0+91)];
  var val120 = data1_65536[(alu0+92)];
  var val121 = data1_65536[(alu0+93)];
  var val122 = data1_65536[(alu0+94)];
  var val123 = data1_65536[(alu0+95)];
  var val124 = data1_65536[(alu0+96)];
  var val125 = data1_65536[(alu0+97)];
  var val126 = data1_65536[(alu0+98)];
  var val127 = data1_65536[(alu0+99)];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast4 = bitcast<u32>(gidx0);
  var cast5 = bitcast<u32>(lidx1);
  var alu1 = (bitcast<i32>((cast4<<11u))+bitcast<i32>((cast5<<7u))+cast2);
  var val128 = data2_65536[alu1];
  var val129 = data2_65536[(alu1+1)];
  var val130 = data2_65536[(alu1+10)];
  var val131 = data2_65536[(alu1+100)];
  var val132 = data2_65536[(alu1+101)];
  var val133 = data2_65536[(alu1+102)];
  var val134 = data2_65536[(alu1+103)];
  var val135 = data2_65536[(alu1+104)];
  var val136 = data2_65536[(alu1+105)];
  var val137 = data2_65536[(alu1+106)];
  var val138 = data2_65536[(alu1+107)];
  var val139 = data2_65536[(alu1+108)];
  var val140 = data2_65536[(alu1+109)];
  var val141 = data2_65536[(alu1+11)];
  var val142 = data2_65536[(alu1+110)];
  var val143 = data2_65536[(alu1+111)];
  var val144 = data2_65536[(alu1+112)];
  var val145 = data2_65536[(alu1+113)];
  var val146 = data2_65536[(alu1+114)];
  var val147 = data2_65536[(alu1+115)];
  var val148 = data2_65536[(alu1+116)];
  var val149 = data2_65536[(alu1+117)];
  var val150 = data2_65536[(alu1+118)];
  var val151 = data2_65536[(alu1+119)];
  var val152 = data2_65536[(alu1+12)];
  var val153 = data2_65536[(alu1+120)];
  var val154 = data2_65536[(alu1+121)];
  var val155 = data2_65536[(alu1+122)];
  var val156 = data2_65536[(alu1+123)];
  var val157 = data2_65536[(alu1+124)];
  var val158 = data2_65536[(alu1+125)];
  var val159 = data2_65536[(alu1+126)];
  var val160 = data2_65536[(alu1+127)];
  var val161 = data2_65536[(alu1+13)];
  var val162 = data2_65536[(alu1+14)];
  var val163 = data2_65536[(alu1+15)];
  var val164 = data2_65536[(alu1+16)];
  var val165 = data2_65536[(alu1+17)];
  var val166 = data2_65536[(alu1+18)];
  var val167 = data2_65536[(alu1+19)];
  var val168 = data2_65536[(alu1+2)];
  var val169 = data2_65536[(alu1+20)];
  var val170 = data2_65536[(alu1+21)];
  var val171 = data2_65536[(alu1+22)];
  var val172 = data2_65536[(alu1+23)];
  var val173 = data2_65536[(alu1+24)];
  var val174 = data2_65536[(alu1+25)];
  var val175 = data2_65536[(alu1+26)];
  var val176 = data2_65536[(alu1+27)];
  var val177 = data2_65536[(alu1+28)];
  var val178 = data2_65536[(alu1+29)];
  var val179 = data2_65536[(alu1+3)];
  var val180 = data2_65536[(alu1+30)];
  var val181 = data2_65536[(alu1+31)];
  var val182 = data2_65536[(alu1+32)];
  var val183 = data2_65536[(alu1+33)];
  var val184 = data2_65536[(alu1+34)];
  var val185 = data2_65536[(alu1+35)];
  var val186 = data2_65536[(alu1+36)];
  var val187 = data2_65536[(alu1+37)];
  var val188 = data2_65536[(alu1+38)];
  var val189 = data2_65536[(alu1+39)];
  var val190 = data2_65536[(alu1+4)];
  var val191 = data2_65536[(alu1+40)];
  var val192 = data2_65536[(alu1+41)];
  var val193 = data2_65536[(alu1+42)];
  var val194 = data2_65536[(alu1+43)];
  var val195 = data2_65536[(alu1+44)];
  var val196 = data2_65536[(alu1+45)];
  var val197 = data2_65536[(alu1+46)];
  var val198 = data2_65536[(alu1+47)];
  var val199 = data2_65536[(alu1+48)];
  var val200 = data2_65536[(alu1+49)];
  var val201 = data2_65536[(alu1+5)];
  var val202 = data2_65536[(alu1+50)];
  var val203 = data2_65536[(alu1+51)];
  var val204 = data2_65536[(alu1+52)];
  var val205 = data2_65536[(alu1+53)];
  var val206 = data2_65536[(alu1+54)];
  var val207 = data2_65536[(alu1+55)];
  var val208 = data2_65536[(alu1+56)];
  var val209 = data2_65536[(alu1+57)];
  var val210 = data2_65536[(alu1+58)];
  var val211 = data2_65536[(alu1+59)];
  var val212 = data2_65536[(alu1+6)];
  var val213 = data2_65536[(alu1+60)];
  var val214 = data2_65536[(alu1+61)];
  var val215 = data2_65536[(alu1+62)];
  var val216 = data2_65536[(alu1+63)];
  var val217 = data2_65536[(alu1+64)];
  var val218 = data2_65536[(alu1+65)];
  var val219 = data2_65536[(alu1+66)];
  var val220 = data2_65536[(alu1+67)];
  var val221 = data2_65536[(alu1+68)];
  var val222 = data2_65536[(alu1+69)];
  var val223 = data2_65536[(alu1+7)];
  var val224 = data2_65536[(alu1+70)];
  var val225 = data2_65536[(alu1+71)];
  var val226 = data2_65536[(alu1+72)];
  var val227 = data2_65536[(alu1+73)];
  var val228 = data2_65536[(alu1+74)];
  var val229 = data2_65536[(alu1+75)];
  var val230 = data2_65536[(alu1+76)];
  var val231 = data2_65536[(alu1+77)];
  var val232 = data2_65536[(alu1+78)];
  var val233 = data2_65536[(alu1+79)];
  var val234 = data2_65536[(alu1+8)];
  var val235 = data2_65536[(alu1+80)];
  var val236 = data2_65536[(alu1+81)];
  var val237 = data2_65536[(alu1+82)];
  var val238 = data2_65536[(alu1+83)];
  var val239 = data2_65536[(alu1+84)];
  var val240 = data2_65536[(alu1+85)];
  var val241 = data2_65536[(alu1+86)];
  var val242 = data2_65536[(alu1+87)];
  var val243 = data2_65536[(alu1+88)];
  var val244 = data2_65536[(alu1+89)];
  var val245 = data2_65536[(alu1+9)];
  var val246 = data2_65536[(alu1+90)];
  var val247 = data2_65536[(alu1+91)];
  var val248 = data2_65536[(alu1+92)];
  var val249 = data2_65536[(alu1+93)];
  var val250 = data2_65536[(alu1+94)];
  var val251 = data2_65536[(alu1+95)];
  var val252 = data2_65536[(alu1+96)];
  var val253 = data2_65536[(alu1+97)];
  var val254 = data2_65536[(alu1+98)];
  var val255 = data2_65536[(alu1+99)];
  var alu2 = (bitcast<i32>((cast4<<6u))+bitcast<i32>((cast5<<2u))+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast1<<16u)));
  data0_524288[alu2] = (((val0*val128)+(val1*val129)+(val40*val168)+(val51*val179)+(val62*val190)+(val73*val201)+(val84*val212)+(val95*val223)+(val106*val234)+(val117*val245)+(val2*val130)+(val13*val141)+(val24*val152)+(val33*val161)+(val34*val162)+(val35*val163)+(val36*val164)+(val37*val165)+(val38*val166)+(val39*val167)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val173)+(val46*val174)+(val47*val175)+(val48*val176)+(val49*val177)+(val50*val178)+(val52*val180)+(val53*val181))*0.1767766922712326f);
  data0_524288[(alu2+1)] = (((val0*val182)+(val1*val183)+(val40*val184)+(val51*val185)+(val62*val186)+(val73*val187)+(val84*val188)+(val95*val189)+(val106*val191)+(val117*val192)+(val2*val193)+(val13*val194)+(val24*val195)+(val33*val196)+(val34*val197)+(val35*val198)+(val36*val199)+(val37*val200)+(val38*val202)+(val39*val203)+(val41*val204)+(val42*val205)+(val43*val206)+(val44*val207)+(val45*val208)+(val46*val209)+(val47*val210)+(val48*val211)+(val49*val213)+(val50*val214)+(val52*val215)+(val53*val216))*0.1767766922712326f);
  data0_524288[(alu2+2)] = (((val0*val217)+(val1*val218)+(val40*val219)+(val51*val220)+(val62*val221)+(val73*val222)+(val84*val224)+(val95*val225)+(val106*val226)+(val117*val227)+(val2*val228)+(val13*val229)+(val24*val230)+(val33*val231)+(val34*val232)+(val35*val233)+(val36*val235)+(val37*val236)+(val38*val237)+(val39*val238)+(val41*val239)+(val42*val240)+(val43*val241)+(val44*val242)+(val45*val243)+(val46*val244)+(val47*val246)+(val48*val247)+(val49*val248)+(val50*val249)+(val52*val250)+(val53*val251))*0.1767766922712326f);
  data0_524288[(alu2+256)] = (((val54*val128)+(val55*val129)+(val56*val168)+(val57*val179)+(val58*val190)+(val59*val201)+(val60*val212)+(val61*val223)+(val63*val234)+(val64*val245)+(val65*val130)+(val66*val141)+(val67*val152)+(val68*val161)+(val69*val162)+(val70*val163)+(val71*val164)+(val72*val165)+(val74*val166)+(val75*val167)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val173)+(val81*val174)+(val82*val175)+(val83*val176)+(val85*val177)+(val86*val178)+(val87*val180)+(val88*val181))*0.1767766922712326f);
  data0_524288[(alu2+257)] = (((val54*val182)+(val55*val183)+(val56*val184)+(val57*val185)+(val58*val186)+(val59*val187)+(val60*val188)+(val61*val189)+(val63*val191)+(val64*val192)+(val65*val193)+(val66*val194)+(val67*val195)+(val68*val196)+(val69*val197)+(val70*val198)+(val71*val199)+(val72*val200)+(val74*val202)+(val75*val203)+(val76*val204)+(val77*val205)+(val78*val206)+(val79*val207)+(val80*val208)+(val81*val209)+(val82*val210)+(val83*val211)+(val85*val213)+(val86*val214)+(val87*val215)+(val88*val216))*0.1767766922712326f);
  data0_524288[(alu2+258)] = (((val54*val217)+(val55*val218)+(val56*val219)+(val57*val220)+(val58*val221)+(val59*val222)+(val60*val224)+(val61*val225)+(val63*val226)+(val64*val227)+(val65*val228)+(val66*val229)+(val67*val230)+(val68*val231)+(val69*val232)+(val70*val233)+(val71*val235)+(val72*val236)+(val74*val237)+(val75*val238)+(val76*val239)+(val77*val240)+(val78*val241)+(val79*val242)+(val80*val243)+(val81*val244)+(val82*val246)+(val83*val247)+(val85*val248)+(val86*val249)+(val87*val250)+(val88*val251))*0.1767766922712326f);
  data0_524288[(alu2+259)] = (((val54*val252)+(val55*val253)+(val56*val254)+(val57*val255)+(val58*val131)+(val59*val132)+(val60*val133)+(val61*val134)+(val63*val135)+(val64*val136)+(val65*val137)+(val66*val138)+(val67*val139)+(val68*val140)+(val69*val142)+(val70*val143)+(val71*val144)+(val72*val145)+(val74*val146)+(val75*val147)+(val76*val148)+(val77*val149)+(val78*val150)+(val79*val151)+(val80*val153)+(val81*val154)+(val82*val155)+(val83*val156)+(val85*val157)+(val86*val158)+(val87*val159)+(val88*val160))*0.1767766922712326f);
  data0_524288[(alu2+3)] = (((val0*val252)+(val1*val253)+(val40*val254)+(val51*val255)+(val62*val131)+(val73*val132)+(val84*val133)+(val95*val134)+(val106*val135)+(val117*val136)+(val2*val137)+(val13*val138)+(val24*val139)+(val33*val140)+(val34*val142)+(val35*val143)+(val36*val144)+(val37*val145)+(val38*val146)+(val39*val147)+(val41*val148)+(val42*val149)+(val43*val150)+(val44*val151)+(val45*val153)+(val46*val154)+(val47*val155)+(val48*val156)+(val49*val157)+(val50*val158)+(val52*val159)+(val53*val160))*0.1767766922712326f);
  data0_524288[(alu2+512)] = (((val89*val128)+(val90*val129)+(val91*val168)+(val92*val179)+(val93*val190)+(val94*val201)+(val96*val212)+(val97*val223)+(val98*val234)+(val99*val245)+(val100*val130)+(val101*val141)+(val102*val152)+(val103*val161)+(val104*val162)+(val105*val163)+(val107*val164)+(val108*val165)+(val109*val166)+(val110*val167)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val173)+(val116*val174)+(val118*val175)+(val119*val176)+(val120*val177)+(val121*val178)+(val122*val180)+(val123*val181))*0.1767766922712326f);
  data0_524288[(alu2+513)] = (((val89*val182)+(val90*val183)+(val91*val184)+(val92*val185)+(val93*val186)+(val94*val187)+(val96*val188)+(val97*val189)+(val98*val191)+(val99*val192)+(val100*val193)+(val101*val194)+(val102*val195)+(val103*val196)+(val104*val197)+(val105*val198)+(val107*val199)+(val108*val200)+(val109*val202)+(val110*val203)+(val111*val204)+(val112*val205)+(val113*val206)+(val114*val207)+(val115*val208)+(val116*val209)+(val118*val210)+(val119*val211)+(val120*val213)+(val121*val214)+(val122*val215)+(val123*val216))*0.1767766922712326f);
  data0_524288[(alu2+514)] = (((val89*val217)+(val90*val218)+(val91*val219)+(val92*val220)+(val93*val221)+(val94*val222)+(val96*val224)+(val97*val225)+(val98*val226)+(val99*val227)+(val100*val228)+(val101*val229)+(val102*val230)+(val103*val231)+(val104*val232)+(val105*val233)+(val107*val235)+(val108*val236)+(val109*val237)+(val110*val238)+(val111*val239)+(val112*val240)+(val113*val241)+(val114*val242)+(val115*val243)+(val116*val244)+(val118*val246)+(val119*val247)+(val120*val248)+(val121*val249)+(val122*val250)+(val123*val251))*0.1767766922712326f);
  data0_524288[(alu2+515)] = (((val89*val252)+(val90*val253)+(val91*val254)+(val92*val255)+(val93*val131)+(val94*val132)+(val96*val133)+(val97*val134)+(val98*val135)+(val99*val136)+(val100*val137)+(val101*val138)+(val102*val139)+(val103*val140)+(val104*val142)+(val105*val143)+(val107*val144)+(val108*val145)+(val109*val146)+(val110*val147)+(val111*val148)+(val112*val149)+(val113*val150)+(val114*val151)+(val115*val153)+(val116*val154)+(val118*val155)+(val119*val156)+(val120*val157)+(val121*val158)+(val122*val159)+(val123*val160))*0.1767766922712326f);
  data0_524288[(alu2+768)] = (((val124*val128)+(val125*val129)+(val126*val168)+(val127*val179)+(val3*val190)+(val4*val201)+(val5*val212)+(val6*val223)+(val7*val234)+(val8*val245)+(val9*val130)+(val10*val141)+(val11*val152)+(val12*val161)+(val14*val162)+(val15*val163)+(val16*val164)+(val17*val165)+(val18*val166)+(val19*val167)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val173)+(val26*val174)+(val27*val175)+(val28*val176)+(val29*val177)+(val30*val178)+(val31*val180)+(val32*val181))*0.1767766922712326f);
  data0_524288[(alu2+769)] = (((val124*val182)+(val125*val183)+(val126*val184)+(val127*val185)+(val3*val186)+(val4*val187)+(val5*val188)+(val6*val189)+(val7*val191)+(val8*val192)+(val9*val193)+(val10*val194)+(val11*val195)+(val12*val196)+(val14*val197)+(val15*val198)+(val16*val199)+(val17*val200)+(val18*val202)+(val19*val203)+(val20*val204)+(val21*val205)+(val22*val206)+(val23*val207)+(val25*val208)+(val26*val209)+(val27*val210)+(val28*val211)+(val29*val213)+(val30*val214)+(val31*val215)+(val32*val216))*0.1767766922712326f);
  data0_524288[(alu2+770)] = (((val124*val217)+(val125*val218)+(val126*val219)+(val127*val220)+(val3*val221)+(val4*val222)+(val5*val224)+(val6*val225)+(val7*val226)+(val8*val227)+(val9*val228)+(val10*val229)+(val11*val230)+(val12*val231)+(val14*val232)+(val15*val233)+(val16*val235)+(val17*val236)+(val18*val237)+(val19*val238)+(val20*val239)+(val21*val240)+(val22*val241)+(val23*val242)+(val25*val243)+(val26*val244)+(val27*val246)+(val28*val247)+(val29*val248)+(val30*val249)+(val31*val250)+(val32*val251))*0.1767766922712326f);
  data0_524288[(alu2+771)] = (((val124*val252)+(val125*val253)+(val126*val254)+(val127*val255)+(val3*val131)+(val4*val132)+(val5*val133)+(val6*val134)+(val7*val135)+(val8*val136)+(val9*val137)+(val10*val138)+(val11*val139)+(val12*val140)+(val14*val142)+(val15*val143)+(val16*val144)+(val17*val145)+(val18*val146)+(val19*val147)+(val20*val148)+(val21*val149)+(val22*val150)+(val23*val151)+(val25*val153)+(val26*val154)+(val27*val155)+(val28*val156)+(val29*val157)+(val30*val158)+(val31*val159)+(val32*val160))*0.1767766922712326f);
}`;

const r_2048_16_16_dispatch_101 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = (f32(-INFINITY));
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val0 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    var val1 = buf0[0];
    var alu1 = select(val1,val0,(val1<val0));
    buf0[0] = alu1;
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = (f32(-INFINITY));
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf1[0];
    var val3 = buf2[Lidx102];
    var alu7 = select(val2,val3,(val2<val3));
    buf1[0] = alu7;
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_2048_16_16_dispatch_102 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var val0 = data2_2048[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val1 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    buf0[0] = (buf0[0]+exp2(((val1-val0)*1.4426950216293335f)));
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf2[Lidx102];
    buf1[0] = (buf1[0]+val2);
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_8_4_16_8_4_4_64_4_dispatch_103 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2048:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_196608:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u))+bitcast<i32>((cast1<<8u)));
  var val0 = data2_2048[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_2048[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_2048[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_2048[alu3];
  var lidx1 = i32(lindex.y); /* 8 */
  var cast3 = bitcast<i32>((bitcast<u32>(lidx1)<<2u));
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var alu20 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast2<<10u))+bitcast<i32>((bitcast<u32>(Lidx0)<<2u))+bitcast<i32>((cast1<<16u)));
    var val4 = data1_524288[alu20];
    var val5 = data1_524288[(alu20+1)];
    var val6 = data1_524288[(alu20+2)];
    var val7 = data1_524288[(alu20+256)];
    var val8 = data1_524288[(alu20+257)];
    var val9 = data1_524288[(alu20+258)];
    var val10 = data1_524288[(alu20+259)];
    var val11 = data1_524288[(alu20+3)];
    var val12 = data1_524288[(alu20+512)];
    var val13 = data1_524288[(alu20+513)];
    var val14 = data1_524288[(alu20+514)];
    var val15 = data1_524288[(alu20+515)];
    var val16 = data1_524288[(alu20+768)];
    var val17 = data1_524288[(alu20+769)];
    var val18 = data1_524288[(alu20+770)];
    var val19 = data1_524288[(alu20+771)];
    var alu21 = (bitcast<i32>((cast1<<5u))+cast3+(Lidx0*3072));
    var val20 = data4_196608[(alu21+1280)];
    var val21 = data4_196608[(alu21+1281)];
    var val22 = data4_196608[(alu21+1282)];
    var val23 = data4_196608[(alu21+1283)];
    var val24 = data4_196608[(alu21+2048)];
    var val25 = data4_196608[(alu21+2049)];
    var val26 = data4_196608[(alu21+2050)];
    var val27 = data4_196608[(alu21+2051)];
    var val28 = data4_196608[(alu21+2816)];
    var val29 = data4_196608[(alu21+2817)];
    var val30 = data4_196608[(alu21+2818)];
    var val31 = data4_196608[(alu21+2819)];
    var val32 = data4_196608[(alu21+512)];
    var val33 = data4_196608[(alu21+513)];
    var val34 = data4_196608[(alu21+514)];
    var val35 = data4_196608[(alu21+515)];
    var alu22 = exp2(((val4-val0)*1.4426950216293335f));
    var alu23 = exp2(((val5-val0)*1.4426950216293335f));
    var alu24 = exp2(((val6-val0)*1.4426950216293335f));
    var alu25 = exp2(((val7-val1)*1.4426950216293335f));
    var alu26 = exp2(((val8-val1)*1.4426950216293335f));
    var alu27 = exp2(((val9-val1)*1.4426950216293335f));
    var alu28 = exp2(((val10-val1)*1.4426950216293335f));
    var alu29 = exp2(((val11-val0)*1.4426950216293335f));
    var alu30 = exp2(((val12-val2)*1.4426950216293335f));
    var alu31 = exp2(((val13-val2)*1.4426950216293335f));
    var alu32 = exp2(((val14-val2)*1.4426950216293335f));
    var alu33 = exp2(((val15-val2)*1.4426950216293335f));
    var alu34 = exp2(((val16-val3)*1.4426950216293335f));
    var alu35 = exp2(((val17-val3)*1.4426950216293335f));
    var alu36 = exp2(((val18-val3)*1.4426950216293335f));
    var alu37 = exp2(((val19-val3)*1.4426950216293335f));
    buf0[0] = (buf0[0]+(alu22*val32)+(alu23*val20)+(alu24*val24)+(alu29*val28));
    buf0[1] = (buf0[1]+(alu22*val33)+(alu23*val21)+(alu24*val25)+(alu29*val29));
    buf0[10] = (buf0[10]+(alu30*val34)+(alu31*val22)+(alu32*val26)+(alu33*val30));
    buf0[11] = (buf0[11]+(alu30*val35)+(alu31*val23)+(alu32*val27)+(alu33*val31));
    buf0[12] = (buf0[12]+(alu34*val32)+(alu35*val20)+(alu36*val24)+(alu37*val28));
    buf0[13] = (buf0[13]+(alu34*val33)+(alu35*val21)+(alu36*val25)+(alu37*val29));
    buf0[14] = (buf0[14]+(alu34*val34)+(alu35*val22)+(alu36*val26)+(alu37*val30));
    buf0[15] = (buf0[15]+(alu34*val35)+(alu35*val23)+(alu36*val27)+(alu37*val31));
    buf0[2] = (buf0[2]+(alu22*val34)+(alu23*val22)+(alu24*val26)+(alu29*val30));
    buf0[3] = (buf0[3]+(alu22*val35)+(alu23*val23)+(alu24*val27)+(alu29*val31));
    buf0[4] = (buf0[4]+(alu25*val32)+(alu26*val20)+(alu27*val24)+(alu28*val28));
    buf0[5] = (buf0[5]+(alu25*val33)+(alu26*val21)+(alu27*val25)+(alu28*val29));
    buf0[6] = (buf0[6]+(alu25*val34)+(alu26*val22)+(alu27*val26)+(alu28*val30));
    buf0[7] = (buf0[7]+(alu25*val35)+(alu26*val23)+(alu27*val27)+(alu28*val31));
    buf0[8] = (buf0[8]+(alu30*val32)+(alu31*val20)+(alu32*val24)+(alu33*val28));
    buf0[9] = (buf0[9]+(alu30*val33)+(alu31*val21)+(alu32*val25)+(alu33*val29));
  }
  var val36 = data3_2048[alu0];
  var val37 = data3_2048[alu1];
  var val38 = data3_2048[alu2];
  var val39 = data3_2048[alu3];
  var alu55 = (bitcast<i32>((cast0<<11u))+bitcast<i32>((cast2<<7u))+cast3+bitcast<i32>((cast1<<13u)));
  var alu56 = (1/val36);
  var alu57 = (1/val37);
  var alu58 = (1/val38);
  var alu59 = (1/val39);
  data0_65536[alu55] = (buf0[0]*alu56);
  data0_65536[(alu55+1)] = (buf0[1]*alu56);
  data0_65536[(alu55+2)] = (buf0[2]*alu56);
  data0_65536[(alu55+3)] = (buf0[3]*alu56);
  data0_65536[(alu55+32)] = (buf0[4]*alu57);
  data0_65536[(alu55+33)] = (buf0[5]*alu57);
  data0_65536[(alu55+34)] = (buf0[6]*alu57);
  data0_65536[(alu55+35)] = (buf0[7]*alu57);
  data0_65536[(alu55+64)] = (buf0[8]*alu58);
  data0_65536[(alu55+65)] = (buf0[9]*alu58);
  data0_65536[(alu55+66)] = (buf0[10]*alu58);
  data0_65536[(alu55+67)] = (buf0[11]*alu58);
  data0_65536[(alu55+96)] = (buf0[12]*alu59);
  data0_65536[(alu55+97)] = (buf0[13]*alu59);
  data0_65536[(alu55+98)] = (buf0[14]*alu59);
  data0_65536[(alu55+99)] = (buf0[15]*alu59);
}`;

const r_8_4_8_16_4_4_8_32_dispatch_104 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_65536:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0_0 = 0; Lidx0_0 < 8; Lidx0_0++) {
    var cast4 = bitcast<u32>(Lidx0_0);
    var alu16 = (bitcast<i32>((cast1<<10u))+bitcast<i32>((cast2<<7u))+bitcast<i32>((cast4<<13u)));
    var val0 = data3_65536[alu16];
    var val1 = data3_65536[(alu16+1)];
    var val2 = data3_65536[(alu16+10)];
    var val3 = data3_65536[(alu16+100)];
    var val4 = data3_65536[(alu16+101)];
    var val5 = data3_65536[(alu16+102)];
    var val6 = data3_65536[(alu16+103)];
    var val7 = data3_65536[(alu16+104)];
    var val8 = data3_65536[(alu16+105)];
    var val9 = data3_65536[(alu16+106)];
    var val10 = data3_65536[(alu16+107)];
    var val11 = data3_65536[(alu16+108)];
    var val12 = data3_65536[(alu16+109)];
    var val13 = data3_65536[(alu16+11)];
    var val14 = data3_65536[(alu16+110)];
    var val15 = data3_65536[(alu16+111)];
    var val16 = data3_65536[(alu16+112)];
    var val17 = data3_65536[(alu16+113)];
    var val18 = data3_65536[(alu16+114)];
    var val19 = data3_65536[(alu16+115)];
    var val20 = data3_65536[(alu16+116)];
    var val21 = data3_65536[(alu16+117)];
    var val22 = data3_65536[(alu16+118)];
    var val23 = data3_65536[(alu16+119)];
    var val24 = data3_65536[(alu16+12)];
    var val25 = data3_65536[(alu16+120)];
    var val26 = data3_65536[(alu16+121)];
    var val27 = data3_65536[(alu16+122)];
    var val28 = data3_65536[(alu16+123)];
    var val29 = data3_65536[(alu16+124)];
    var val30 = data3_65536[(alu16+125)];
    var val31 = data3_65536[(alu16+126)];
    var val32 = data3_65536[(alu16+127)];
    var val33 = data3_65536[(alu16+13)];
    var val34 = data3_65536[(alu16+14)];
    var val35 = data3_65536[(alu16+15)];
    var val36 = data3_65536[(alu16+16)];
    var val37 = data3_65536[(alu16+17)];
    var val38 = data3_65536[(alu16+18)];
    var val39 = data3_65536[(alu16+19)];
    var val40 = data3_65536[(alu16+2)];
    var val41 = data3_65536[(alu16+20)];
    var val42 = data3_65536[(alu16+21)];
    var val43 = data3_65536[(alu16+22)];
    var val44 = data3_65536[(alu16+23)];
    var val45 = data3_65536[(alu16+24)];
    var val46 = data3_65536[(alu16+25)];
    var val47 = data3_65536[(alu16+26)];
    var val48 = data3_65536[(alu16+27)];
    var val49 = data3_65536[(alu16+28)];
    var val50 = data3_65536[(alu16+29)];
    var val51 = data3_65536[(alu16+3)];
    var val52 = data3_65536[(alu16+30)];
    var val53 = data3_65536[(alu16+31)];
    var val54 = data3_65536[(alu16+32)];
    var val55 = data3_65536[(alu16+33)];
    var val56 = data3_65536[(alu16+34)];
    var val57 = data3_65536[(alu16+35)];
    var val58 = data3_65536[(alu16+36)];
    var val59 = data3_65536[(alu16+37)];
    var val60 = data3_65536[(alu16+38)];
    var val61 = data3_65536[(alu16+39)];
    var val62 = data3_65536[(alu16+4)];
    var val63 = data3_65536[(alu16+40)];
    var val64 = data3_65536[(alu16+41)];
    var val65 = data3_65536[(alu16+42)];
    var val66 = data3_65536[(alu16+43)];
    var val67 = data3_65536[(alu16+44)];
    var val68 = data3_65536[(alu16+45)];
    var val69 = data3_65536[(alu16+46)];
    var val70 = data3_65536[(alu16+47)];
    var val71 = data3_65536[(alu16+48)];
    var val72 = data3_65536[(alu16+49)];
    var val73 = data3_65536[(alu16+5)];
    var val74 = data3_65536[(alu16+50)];
    var val75 = data3_65536[(alu16+51)];
    var val76 = data3_65536[(alu16+52)];
    var val77 = data3_65536[(alu16+53)];
    var val78 = data3_65536[(alu16+54)];
    var val79 = data3_65536[(alu16+55)];
    var val80 = data3_65536[(alu16+56)];
    var val81 = data3_65536[(alu16+57)];
    var val82 = data3_65536[(alu16+58)];
    var val83 = data3_65536[(alu16+59)];
    var val84 = data3_65536[(alu16+6)];
    var val85 = data3_65536[(alu16+60)];
    var val86 = data3_65536[(alu16+61)];
    var val87 = data3_65536[(alu16+62)];
    var val88 = data3_65536[(alu16+63)];
    var val89 = data3_65536[(alu16+64)];
    var val90 = data3_65536[(alu16+65)];
    var val91 = data3_65536[(alu16+66)];
    var val92 = data3_65536[(alu16+67)];
    var val93 = data3_65536[(alu16+68)];
    var val94 = data3_65536[(alu16+69)];
    var val95 = data3_65536[(alu16+7)];
    var val96 = data3_65536[(alu16+70)];
    var val97 = data3_65536[(alu16+71)];
    var val98 = data3_65536[(alu16+72)];
    var val99 = data3_65536[(alu16+73)];
    var val100 = data3_65536[(alu16+74)];
    var val101 = data3_65536[(alu16+75)];
    var val102 = data3_65536[(alu16+76)];
    var val103 = data3_65536[(alu16+77)];
    var val104 = data3_65536[(alu16+78)];
    var val105 = data3_65536[(alu16+79)];
    var val106 = data3_65536[(alu16+8)];
    var val107 = data3_65536[(alu16+80)];
    var val108 = data3_65536[(alu16+81)];
    var val109 = data3_65536[(alu16+82)];
    var val110 = data3_65536[(alu16+83)];
    var val111 = data3_65536[(alu16+84)];
    var val112 = data3_65536[(alu16+85)];
    var val113 = data3_65536[(alu16+86)];
    var val114 = data3_65536[(alu16+87)];
    var val115 = data3_65536[(alu16+88)];
    var val116 = data3_65536[(alu16+89)];
    var val117 = data3_65536[(alu16+9)];
    var val118 = data3_65536[(alu16+90)];
    var val119 = data3_65536[(alu16+91)];
    var val120 = data3_65536[(alu16+92)];
    var val121 = data3_65536[(alu16+93)];
    var val122 = data3_65536[(alu16+94)];
    var val123 = data3_65536[(alu16+95)];
    var val124 = data3_65536[(alu16+96)];
    var val125 = data3_65536[(alu16+97)];
    var val126 = data3_65536[(alu16+98)];
    var val127 = data3_65536[(alu16+99)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast4<<5u)));
    var val128 = data4_65536[alu17];
    var val129 = data4_65536[(alu17+1)];
    var val130 = data4_65536[(alu17+10)];
    var val131 = data4_65536[(alu17+11)];
    var val132 = data4_65536[(alu17+12)];
    var val133 = data4_65536[(alu17+13)];
    var val134 = data4_65536[(alu17+14)];
    var val135 = data4_65536[(alu17+15)];
    var val136 = data4_65536[(alu17+16)];
    var val137 = data4_65536[(alu17+17)];
    var val138 = data4_65536[(alu17+18)];
    var val139 = data4_65536[(alu17+19)];
    var val140 = data4_65536[(alu17+2)];
    var val141 = data4_65536[(alu17+20)];
    var val142 = data4_65536[(alu17+21)];
    var val143 = data4_65536[(alu17+22)];
    var val144 = data4_65536[(alu17+23)];
    var val145 = data4_65536[(alu17+24)];
    var val146 = data4_65536[(alu17+25)];
    var val147 = data4_65536[(alu17+256)];
    var val148 = data4_65536[(alu17+257)];
    var val149 = data4_65536[(alu17+258)];
    var val150 = data4_65536[(alu17+259)];
    var val151 = data4_65536[(alu17+26)];
    var val152 = data4_65536[(alu17+260)];
    var val153 = data4_65536[(alu17+261)];
    var val154 = data4_65536[(alu17+262)];
    var val155 = data4_65536[(alu17+263)];
    var val156 = data4_65536[(alu17+264)];
    var val157 = data4_65536[(alu17+265)];
    var val158 = data4_65536[(alu17+266)];
    var val159 = data4_65536[(alu17+267)];
    var val160 = data4_65536[(alu17+268)];
    var val161 = data4_65536[(alu17+269)];
    var val162 = data4_65536[(alu17+27)];
    var val163 = data4_65536[(alu17+270)];
    var val164 = data4_65536[(alu17+271)];
    var val165 = data4_65536[(alu17+272)];
    var val166 = data4_65536[(alu17+273)];
    var val167 = data4_65536[(alu17+274)];
    var val168 = data4_65536[(alu17+275)];
    var val169 = data4_65536[(alu17+276)];
    var val170 = data4_65536[(alu17+277)];
    var val171 = data4_65536[(alu17+278)];
    var val172 = data4_65536[(alu17+279)];
    var val173 = data4_65536[(alu17+28)];
    var val174 = data4_65536[(alu17+280)];
    var val175 = data4_65536[(alu17+281)];
    var val176 = data4_65536[(alu17+282)];
    var val177 = data4_65536[(alu17+283)];
    var val178 = data4_65536[(alu17+284)];
    var val179 = data4_65536[(alu17+285)];
    var val180 = data4_65536[(alu17+286)];
    var val181 = data4_65536[(alu17+287)];
    var val182 = data4_65536[(alu17+29)];
    var val183 = data4_65536[(alu17+3)];
    var val184 = data4_65536[(alu17+30)];
    var val185 = data4_65536[(alu17+31)];
    var val186 = data4_65536[(alu17+4)];
    var val187 = data4_65536[(alu17+5)];
    var val188 = data4_65536[(alu17+512)];
    var val189 = data4_65536[(alu17+513)];
    var val190 = data4_65536[(alu17+514)];
    var val191 = data4_65536[(alu17+515)];
    var val192 = data4_65536[(alu17+516)];
    var val193 = data4_65536[(alu17+517)];
    var val194 = data4_65536[(alu17+518)];
    var val195 = data4_65536[(alu17+519)];
    var val196 = data4_65536[(alu17+520)];
    var val197 = data4_65536[(alu17+521)];
    var val198 = data4_65536[(alu17+522)];
    var val199 = data4_65536[(alu17+523)];
    var val200 = data4_65536[(alu17+524)];
    var val201 = data4_65536[(alu17+525)];
    var val202 = data4_65536[(alu17+526)];
    var val203 = data4_65536[(alu17+527)];
    var val204 = data4_65536[(alu17+528)];
    var val205 = data4_65536[(alu17+529)];
    var val206 = data4_65536[(alu17+530)];
    var val207 = data4_65536[(alu17+531)];
    var val208 = data4_65536[(alu17+532)];
    var val209 = data4_65536[(alu17+533)];
    var val210 = data4_65536[(alu17+534)];
    var val211 = data4_65536[(alu17+535)];
    var val212 = data4_65536[(alu17+536)];
    var val213 = data4_65536[(alu17+537)];
    var val214 = data4_65536[(alu17+538)];
    var val215 = data4_65536[(alu17+539)];
    var val216 = data4_65536[(alu17+540)];
    var val217 = data4_65536[(alu17+541)];
    var val218 = data4_65536[(alu17+542)];
    var val219 = data4_65536[(alu17+543)];
    var val220 = data4_65536[(alu17+6)];
    var val221 = data4_65536[(alu17+7)];
    var val222 = data4_65536[(alu17+768)];
    var val223 = data4_65536[(alu17+769)];
    var val224 = data4_65536[(alu17+770)];
    var val225 = data4_65536[(alu17+771)];
    var val226 = data4_65536[(alu17+772)];
    var val227 = data4_65536[(alu17+773)];
    var val228 = data4_65536[(alu17+774)];
    var val229 = data4_65536[(alu17+775)];
    var val230 = data4_65536[(alu17+776)];
    var val231 = data4_65536[(alu17+777)];
    var val232 = data4_65536[(alu17+778)];
    var val233 = data4_65536[(alu17+779)];
    var val234 = data4_65536[(alu17+780)];
    var val235 = data4_65536[(alu17+781)];
    var val236 = data4_65536[(alu17+782)];
    var val237 = data4_65536[(alu17+783)];
    var val238 = data4_65536[(alu17+784)];
    var val239 = data4_65536[(alu17+785)];
    var val240 = data4_65536[(alu17+786)];
    var val241 = data4_65536[(alu17+787)];
    var val242 = data4_65536[(alu17+788)];
    var val243 = data4_65536[(alu17+789)];
    var val244 = data4_65536[(alu17+790)];
    var val245 = data4_65536[(alu17+791)];
    var val246 = data4_65536[(alu17+792)];
    var val247 = data4_65536[(alu17+793)];
    var val248 = data4_65536[(alu17+794)];
    var val249 = data4_65536[(alu17+795)];
    var val250 = data4_65536[(alu17+796)];
    var val251 = data4_65536[(alu17+797)];
    var val252 = data4_65536[(alu17+798)];
    var val253 = data4_65536[(alu17+799)];
    var val254 = data4_65536[(alu17+8)];
    var val255 = data4_65536[(alu17+9)];
    buf0[0] = (buf0[0]+(val0*val128)+(val1*val129)+(val40*val140)+(val51*val183)+(val62*val186)+(val73*val187)+(val84*val220)+(val95*val221)+(val106*val254)+(val117*val255)+(val2*val130)+(val13*val131)+(val24*val132)+(val33*val133)+(val34*val134)+(val35*val135)+(val36*val136)+(val37*val137)+(val38*val138)+(val39*val139)+(val41*val141)+(val42*val142)+(val43*val143)+(val44*val144)+(val45*val145)+(val46*val146)+(val47*val151)+(val48*val162)+(val49*val173)+(val50*val182)+(val52*val184)+(val53*val185));
    buf0[1] = (buf0[1]+(val0*val147)+(val1*val148)+(val40*val149)+(val51*val150)+(val62*val152)+(val73*val153)+(val84*val154)+(val95*val155)+(val106*val156)+(val117*val157)+(val2*val158)+(val13*val159)+(val24*val160)+(val33*val161)+(val34*val163)+(val35*val164)+(val36*val165)+(val37*val166)+(val38*val167)+(val39*val168)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val174)+(val46*val175)+(val47*val176)+(val48*val177)+(val49*val178)+(val50*val179)+(val52*val180)+(val53*val181));
    buf0[10] = (buf0[10]+(val89*val188)+(val90*val189)+(val91*val190)+(val92*val191)+(val93*val192)+(val94*val193)+(val96*val194)+(val97*val195)+(val98*val196)+(val99*val197)+(val100*val198)+(val101*val199)+(val102*val200)+(val103*val201)+(val104*val202)+(val105*val203)+(val107*val204)+(val108*val205)+(val109*val206)+(val110*val207)+(val111*val208)+(val112*val209)+(val113*val210)+(val114*val211)+(val115*val212)+(val116*val213)+(val118*val214)+(val119*val215)+(val120*val216)+(val121*val217)+(val122*val218)+(val123*val219));
    buf0[11] = (buf0[11]+(val89*val222)+(val90*val223)+(val91*val224)+(val92*val225)+(val93*val226)+(val94*val227)+(val96*val228)+(val97*val229)+(val98*val230)+(val99*val231)+(val100*val232)+(val101*val233)+(val102*val234)+(val103*val235)+(val104*val236)+(val105*val237)+(val107*val238)+(val108*val239)+(val109*val240)+(val110*val241)+(val111*val242)+(val112*val243)+(val113*val244)+(val114*val245)+(val115*val246)+(val116*val247)+(val118*val248)+(val119*val249)+(val120*val250)+(val121*val251)+(val122*val252)+(val123*val253));
    buf0[12] = (buf0[12]+(val124*val128)+(val125*val129)+(val126*val140)+(val127*val183)+(val3*val186)+(val4*val187)+(val5*val220)+(val6*val221)+(val7*val254)+(val8*val255)+(val9*val130)+(val10*val131)+(val11*val132)+(val12*val133)+(val14*val134)+(val15*val135)+(val16*val136)+(val17*val137)+(val18*val138)+(val19*val139)+(val20*val141)+(val21*val142)+(val22*val143)+(val23*val144)+(val25*val145)+(val26*val146)+(val27*val151)+(val28*val162)+(val29*val173)+(val30*val182)+(val31*val184)+(val32*val185));
    buf0[13] = (buf0[13]+(val124*val147)+(val125*val148)+(val126*val149)+(val127*val150)+(val3*val152)+(val4*val153)+(val5*val154)+(val6*val155)+(val7*val156)+(val8*val157)+(val9*val158)+(val10*val159)+(val11*val160)+(val12*val161)+(val14*val163)+(val15*val164)+(val16*val165)+(val17*val166)+(val18*val167)+(val19*val168)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val174)+(val26*val175)+(val27*val176)+(val28*val177)+(val29*val178)+(val30*val179)+(val31*val180)+(val32*val181));
    buf0[14] = (buf0[14]+(val124*val188)+(val125*val189)+(val126*val190)+(val127*val191)+(val3*val192)+(val4*val193)+(val5*val194)+(val6*val195)+(val7*val196)+(val8*val197)+(val9*val198)+(val10*val199)+(val11*val200)+(val12*val201)+(val14*val202)+(val15*val203)+(val16*val204)+(val17*val205)+(val18*val206)+(val19*val207)+(val20*val208)+(val21*val209)+(val22*val210)+(val23*val211)+(val25*val212)+(val26*val213)+(val27*val214)+(val28*val215)+(val29*val216)+(val30*val217)+(val31*val218)+(val32*val219));
    buf0[15] = (buf0[15]+(val124*val222)+(val125*val223)+(val126*val224)+(val127*val225)+(val3*val226)+(val4*val227)+(val5*val228)+(val6*val229)+(val7*val230)+(val8*val231)+(val9*val232)+(val10*val233)+(val11*val234)+(val12*val235)+(val14*val236)+(val15*val237)+(val16*val238)+(val17*val239)+(val18*val240)+(val19*val241)+(val20*val242)+(val21*val243)+(val22*val244)+(val23*val245)+(val25*val246)+(val26*val247)+(val27*val248)+(val28*val249)+(val29*val250)+(val30*val251)+(val31*val252)+(val32*val253));
    buf0[2] = (buf0[2]+(val0*val188)+(val1*val189)+(val40*val190)+(val51*val191)+(val62*val192)+(val73*val193)+(val84*val194)+(val95*val195)+(val106*val196)+(val117*val197)+(val2*val198)+(val13*val199)+(val24*val200)+(val33*val201)+(val34*val202)+(val35*val203)+(val36*val204)+(val37*val205)+(val38*val206)+(val39*val207)+(val41*val208)+(val42*val209)+(val43*val210)+(val44*val211)+(val45*val212)+(val46*val213)+(val47*val214)+(val48*val215)+(val49*val216)+(val50*val217)+(val52*val218)+(val53*val219));
    buf0[3] = (buf0[3]+(val0*val222)+(val1*val223)+(val40*val224)+(val51*val225)+(val62*val226)+(val73*val227)+(val84*val228)+(val95*val229)+(val106*val230)+(val117*val231)+(val2*val232)+(val13*val233)+(val24*val234)+(val33*val235)+(val34*val236)+(val35*val237)+(val36*val238)+(val37*val239)+(val38*val240)+(val39*val241)+(val41*val242)+(val42*val243)+(val43*val244)+(val44*val245)+(val45*val246)+(val46*val247)+(val47*val248)+(val48*val249)+(val49*val250)+(val50*val251)+(val52*val252)+(val53*val253));
    buf0[4] = (buf0[4]+(val54*val128)+(val55*val129)+(val56*val140)+(val57*val183)+(val58*val186)+(val59*val187)+(val60*val220)+(val61*val221)+(val63*val254)+(val64*val255)+(val65*val130)+(val66*val131)+(val67*val132)+(val68*val133)+(val69*val134)+(val70*val135)+(val71*val136)+(val72*val137)+(val74*val138)+(val75*val139)+(val76*val141)+(val77*val142)+(val78*val143)+(val79*val144)+(val80*val145)+(val81*val146)+(val82*val151)+(val83*val162)+(val85*val173)+(val86*val182)+(val87*val184)+(val88*val185));
    buf0[5] = (buf0[5]+(val54*val147)+(val55*val148)+(val56*val149)+(val57*val150)+(val58*val152)+(val59*val153)+(val60*val154)+(val61*val155)+(val63*val156)+(val64*val157)+(val65*val158)+(val66*val159)+(val67*val160)+(val68*val161)+(val69*val163)+(val70*val164)+(val71*val165)+(val72*val166)+(val74*val167)+(val75*val168)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val174)+(val81*val175)+(val82*val176)+(val83*val177)+(val85*val178)+(val86*val179)+(val87*val180)+(val88*val181));
    buf0[6] = (buf0[6]+(val54*val188)+(val55*val189)+(val56*val190)+(val57*val191)+(val58*val192)+(val59*val193)+(val60*val194)+(val61*val195)+(val63*val196)+(val64*val197)+(val65*val198)+(val66*val199)+(val67*val200)+(val68*val201)+(val69*val202)+(val70*val203)+(val71*val204)+(val72*val205)+(val74*val206)+(val75*val207)+(val76*val208)+(val77*val209)+(val78*val210)+(val79*val211)+(val80*val212)+(val81*val213)+(val82*val214)+(val83*val215)+(val85*val216)+(val86*val217)+(val87*val218)+(val88*val219));
    buf0[7] = (buf0[7]+(val54*val222)+(val55*val223)+(val56*val224)+(val57*val225)+(val58*val226)+(val59*val227)+(val60*val228)+(val61*val229)+(val63*val230)+(val64*val231)+(val65*val232)+(val66*val233)+(val67*val234)+(val68*val235)+(val69*val236)+(val70*val237)+(val71*val238)+(val72*val239)+(val74*val240)+(val75*val241)+(val76*val242)+(val77*val243)+(val78*val244)+(val79*val245)+(val80*val246)+(val81*val247)+(val82*val248)+(val83*val249)+(val85*val250)+(val86*val251)+(val87*val252)+(val88*val253));
    buf0[8] = (buf0[8]+(val89*val128)+(val90*val129)+(val91*val140)+(val92*val183)+(val93*val186)+(val94*val187)+(val96*val220)+(val97*val221)+(val98*val254)+(val99*val255)+(val100*val130)+(val101*val131)+(val102*val132)+(val103*val133)+(val104*val134)+(val105*val135)+(val107*val136)+(val108*val137)+(val109*val138)+(val110*val139)+(val111*val141)+(val112*val142)+(val113*val143)+(val114*val144)+(val115*val145)+(val116*val146)+(val118*val151)+(val119*val162)+(val120*val173)+(val121*val182)+(val122*val184)+(val123*val185));
    buf0[9] = (buf0[9]+(val89*val147)+(val90*val148)+(val91*val149)+(val92*val150)+(val93*val152)+(val94*val153)+(val96*val154)+(val97*val155)+(val98*val156)+(val99*val157)+(val100*val158)+(val101*val159)+(val102*val160)+(val103*val161)+(val104*val163)+(val105*val164)+(val107*val165)+(val108*val166)+(val109*val167)+(val110*val168)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val174)+(val116*val175)+(val118*val176)+(val119*val177)+(val120*val178)+(val121*val179)+(val122*val180)+(val123*val181));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<13u))+bitcast<i32>((cast2<<10u)));
  var val256 = data1_65536[alu36];
  var alu37 = (alu36+1);
  var val257 = data1_65536[alu37];
  var alu38 = (alu36+2);
  var val258 = data1_65536[alu38];
  var alu39 = (alu36+256);
  var val259 = data1_65536[alu39];
  var alu40 = (alu36+257);
  var val260 = data1_65536[alu40];
  var alu41 = (alu36+258);
  var val261 = data1_65536[alu41];
  var alu42 = (alu36+259);
  var val262 = data1_65536[alu42];
  var alu43 = (alu36+3);
  var val263 = data1_65536[alu43];
  var alu44 = (alu36+512);
  var val264 = data1_65536[alu44];
  var alu45 = (alu36+513);
  var val265 = data1_65536[alu45];
  var alu46 = (alu36+514);
  var val266 = data1_65536[alu46];
  var alu47 = (alu36+515);
  var val267 = data1_65536[alu47];
  var alu48 = (alu36+768);
  var val268 = data1_65536[alu48];
  var alu49 = (alu36+769);
  var val269 = data1_65536[alu49];
  var alu50 = (alu36+770);
  var val270 = data1_65536[alu50];
  var alu51 = (alu36+771);
  var val271 = data1_65536[alu51];
  var val272 = data2_1536[(alu35+512)];
  var val273 = data2_1536[(alu35+513)];
  var val274 = data2_1536[(alu35+514)];
  var val275 = data2_1536[(alu35+515)];
  var val276 = data5_256[alu35];
  var val277 = data5_256[(alu35+1)];
  var val278 = data5_256[(alu35+2)];
  var val279 = data5_256[(alu35+3)];
  data0_65536[alu36] = (val256+(val272*(buf0[0]+val276)));
  data0_65536[alu37] = (val257+(val273*(buf0[1]+val277)));
  data0_65536[alu38] = (val258+(val274*(buf0[2]+val278)));
  data0_65536[alu39] = (val259+(val272*(buf0[4]+val276)));
  data0_65536[alu40] = (val260+(val273*(buf0[5]+val277)));
  data0_65536[alu41] = (val261+(val274*(buf0[6]+val278)));
  data0_65536[alu42] = (val262+(val275*(buf0[7]+val279)));
  data0_65536[alu43] = (val263+(val275*(buf0[3]+val279)));
  data0_65536[alu44] = (val264+(val272*(buf0[8]+val276)));
  data0_65536[alu45] = (val265+(val273*(buf0[9]+val277)));
  data0_65536[alu46] = (val266+(val274*(buf0[10]+val278)));
  data0_65536[alu47] = (val267+(val275*(buf0[11]+val279)));
  data0_65536[alu48] = (val268+(val272*(buf0[12]+val276)));
  data0_65536[alu49] = (val269+(val273*(buf0[13]+val277)));
  data0_65536[alu50] = (val270+(val274*(buf0[14]+val278)));
  data0_65536[alu51] = (val271+(val275*(buf0[15]+val279)));
}`;

const r_16_8_4_4_32_dispatch_105 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_106 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var val21 = data3_256[(alu0+1)];
  var val22 = data3_256[(alu0+2)];
  var val23 = data3_256[(alu0+3)];
  var val24 = data4_1536[(alu0+1024)];
  var val25 = data4_1536[(alu0+1025)];
  var val26 = data4_1536[(alu0+1026)];
  var val27 = data4_1536[(alu0+1027)];
  var val28 = data4_1536[(alu0+768)];
  var val29 = data4_1536[(alu0+769)];
  var val30 = data4_1536[(alu0+770)];
  var val31 = data4_1536[(alu0+771)];
  var alu18 = (1/val16);
  var alu19 = (1/val17);
  var alu20 = (1/val18);
  var alu21 = (1/val19);
  var alu22 = (1.0f+val24);
  var alu23 = (1.0f+val25);
  var alu24 = (1.0f+val26);
  var alu25 = (1.0f+val27);
  data0_65536[alu1] = ((val0*alu18*val20*alu22)+val28);
  data0_65536[alu2] = ((val1*alu18*val21*alu23)+val29);
  data0_65536[alu3] = ((val2*alu18*val22*alu24)+val30);
  data0_65536[alu4] = ((val3*alu19*val20*alu22)+val28);
  data0_65536[alu5] = ((val4*alu19*val21*alu23)+val29);
  data0_65536[alu6] = ((val5*alu19*val22*alu24)+val30);
  data0_65536[alu7] = ((val6*alu19*val23*alu25)+val31);
  data0_65536[alu8] = ((val7*alu18*val23*alu25)+val31);
  data0_65536[alu9] = ((val8*alu20*val20*alu22)+val28);
  data0_65536[alu10] = ((val9*alu20*val21*alu23)+val29);
  data0_65536[alu11] = ((val10*alu20*val22*alu24)+val30);
  data0_65536[alu12] = ((val11*alu20*val23*alu25)+val31);
  data0_65536[alu13] = ((val12*alu21*val20*alu22)+val28);
  data0_65536[alu14] = ((val13*alu21*val21*alu23)+val29);
  data0_65536[alu15] = ((val14*alu21*val22*alu24)+val30);
  data0_65536[alu16] = ((val15*alu21*val23*alu25)+val31);
}`;

const r_8_22_8_16_4_4_64_4_dispatch_107 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_360448:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_360448:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1408:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 22 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast2 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast2);
    var val0 = data1_65536[alu16];
    var val1 = data1_65536[(alu16+1)];
    var val2 = data1_65536[(alu16+2)];
    var val3 = data1_65536[(alu16+256)];
    var val4 = data1_65536[(alu16+257)];
    var val5 = data1_65536[(alu16+258)];
    var val6 = data1_65536[(alu16+259)];
    var val7 = data1_65536[(alu16+3)];
    var val8 = data1_65536[(alu16+512)];
    var val9 = data1_65536[(alu16+513)];
    var val10 = data1_65536[(alu16+514)];
    var val11 = data1_65536[(alu16+515)];
    var val12 = data1_65536[(alu16+768)];
    var val13 = data1_65536[(alu16+769)];
    var val14 = data1_65536[(alu16+770)];
    var val15 = data1_65536[(alu16+771)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast1<<10u))+cast2);
    var val16 = data2_360448[alu17];
    var val17 = data2_360448[(alu17+1)];
    var val18 = data2_360448[(alu17+2)];
    var val19 = data2_360448[(alu17+256)];
    var val20 = data2_360448[(alu17+257)];
    var val21 = data2_360448[(alu17+258)];
    var val22 = data2_360448[(alu17+259)];
    var val23 = data2_360448[(alu17+3)];
    var val24 = data2_360448[(alu17+512)];
    var val25 = data2_360448[(alu17+513)];
    var val26 = data2_360448[(alu17+514)];
    var val27 = data2_360448[(alu17+515)];
    var val28 = data2_360448[(alu17+768)];
    var val29 = data2_360448[(alu17+769)];
    var val30 = data2_360448[(alu17+770)];
    var val31 = data2_360448[(alu17+771)];
    buf0[0] = (buf0[0]+(val0*val16)+(val1*val17)+(val2*val18)+(val7*val23));
    buf0[1] = (buf0[1]+(val0*val19)+(val1*val20)+(val2*val21)+(val7*val22));
    buf0[10] = (buf0[10]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    buf0[11] = (buf0[11]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    buf0[12] = (buf0[12]+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val23));
    buf0[13] = (buf0[13]+(val12*val19)+(val13*val20)+(val14*val21)+(val15*val22));
    buf0[14] = (buf0[14]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    buf0[15] = (buf0[15]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    buf0[2] = (buf0[2]+(val0*val24)+(val1*val25)+(val2*val26)+(val7*val27));
    buf0[3] = (buf0[3]+(val0*val28)+(val1*val29)+(val2*val30)+(val7*val31));
    buf0[4] = (buf0[4]+(val3*val16)+(val4*val17)+(val5*val18)+(val6*val23));
    buf0[5] = (buf0[5]+(val3*val19)+(val4*val20)+(val5*val21)+(val6*val22));
    buf0[6] = (buf0[6]+(val3*val24)+(val4*val25)+(val5*val26)+(val6*val27));
    buf0[7] = (buf0[7]+(val3*val28)+(val4*val29)+(val5*val30)+(val6*val31));
    buf0[8] = (buf0[8]+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val23));
    buf0[9] = (buf0[9]+(val8*val19)+(val9*val20)+(val10*val21)+(val11*val22));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)));
  var val32 = data3_1408[alu35];
  var val33 = data3_1408[(alu35+1)];
  var val34 = data3_1408[(alu35+2)];
  var val35 = data3_1408[(alu35+3)];
  var alu36 = (alu35+(gidx1*45056)+(lidx0*5632));
  data0_360448[alu36] = (buf0[0]+val32);
  data0_360448[(alu36+1)] = (buf0[1]+val33);
  data0_360448[(alu36+1408)] = (buf0[4]+val32);
  data0_360448[(alu36+1409)] = (buf0[5]+val33);
  data0_360448[(alu36+1410)] = (buf0[6]+val34);
  data0_360448[(alu36+1411)] = (buf0[7]+val35);
  data0_360448[(alu36+2)] = (buf0[2]+val34);
  data0_360448[(alu36+2816)] = (buf0[8]+val32);
  data0_360448[(alu36+2817)] = (buf0[9]+val33);
  data0_360448[(alu36+2818)] = (buf0[10]+val34);
  data0_360448[(alu36+2819)] = (buf0[11]+val35);
  data0_360448[(alu36+3)] = (buf0[3]+val35);
  data0_360448[(alu36+4224)] = (buf0[12]+val32);
  data0_360448[(alu36+4225)] = (buf0[13]+val33);
  data0_360448[(alu36+4226)] = (buf0[14]+val34);
  data0_360448[(alu36+4227)] = (buf0[15]+val35);
}`;

const r_8_4_8_16_4_4_176_4_dispatch_108 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_360448:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_180224:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 176; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = ((gidx1*45056)+(lidx0*5632)+cast0);
    var val0 = data3_360448[(alu16+1)];
    var val1 = data3_360448[(alu16+1408)];
    var val2 = data3_360448[(alu16+1409)];
    var val3 = data3_360448[(alu16+1410)];
    var val4 = data3_360448[(alu16+1411)];
    var val5 = data3_360448[(alu16+2)];
    var val6 = data3_360448[(alu16+2112)];
    var val7 = data3_360448[(alu16+2113)];
    var val8 = data3_360448[(alu16+2114)];
    var val9 = data3_360448[(alu16+2115)];
    var val10 = data3_360448[(alu16+2816)];
    var val11 = data3_360448[(alu16+2817)];
    var val12 = data3_360448[(alu16+2818)];
    var val13 = data3_360448[(alu16+2819)];
    var val14 = data3_360448[(alu16+3)];
    var val15 = data3_360448[(alu16+3520)];
    var val16 = data3_360448[(alu16+3521)];
    var val17 = data3_360448[(alu16+3522)];
    var val18 = data3_360448[(alu16+3523)];
    var val19 = data3_360448[(alu16+4224)];
    var val20 = data3_360448[(alu16+4225)];
    var val21 = data3_360448[(alu16+4226)];
    var val22 = data3_360448[(alu16+4227)];
    var val23 = data3_360448[(alu16+4928)];
    var val24 = data3_360448[(alu16+4929)];
    var val25 = data3_360448[(alu16+4930)];
    var val26 = data3_360448[(alu16+4931)];
    var val27 = data3_360448[(alu16+704)];
    var val28 = data3_360448[(alu16+705)];
    var val29 = data3_360448[(alu16+706)];
    var val30 = data3_360448[(alu16+707)];
    var val31 = data3_360448[alu16];
    var alu17 = ((gidx0*45056)+(lidx1*2816)+cast0);
    var val32 = data4_180224[(alu17+1)];
    var val33 = data4_180224[(alu17+1408)];
    var val34 = data4_180224[(alu17+1409)];
    var val35 = data4_180224[(alu17+1410)];
    var val36 = data4_180224[(alu17+1411)];
    var val37 = data4_180224[(alu17+2)];
    var val38 = data4_180224[(alu17+2112)];
    var val39 = data4_180224[(alu17+2113)];
    var val40 = data4_180224[(alu17+2114)];
    var val41 = data4_180224[(alu17+2115)];
    var val42 = data4_180224[(alu17+3)];
    var val43 = data4_180224[(alu17+704)];
    var val44 = data4_180224[(alu17+705)];
    var val45 = data4_180224[(alu17+706)];
    var val46 = data4_180224[(alu17+707)];
    var val47 = data4_180224[alu17];
    var alu18 = (val0*(1/(1.0f+exp2((val0*-1.4426950216293335f))))*val28);
    var alu19 = (val31*(1/(1.0f+exp2((val31*-1.4426950216293335f))))*val27);
    var alu20 = (val5*(1/(1.0f+exp2((val5*-1.4426950216293335f))))*val29);
    var alu21 = (val14*(1/(1.0f+exp2((val14*-1.4426950216293335f))))*val30);
    var alu22 = (val10*(1/(1.0f+exp2((val10*-1.4426950216293335f))))*val15);
    var alu23 = (val11*(1/(1.0f+exp2((val11*-1.4426950216293335f))))*val16);
    var alu24 = (val12*(1/(1.0f+exp2((val12*-1.4426950216293335f))))*val17);
    var alu25 = (val13*(1/(1.0f+exp2((val13*-1.4426950216293335f))))*val18);
    var alu26 = (val19*(1/(1.0f+exp2((val19*-1.4426950216293335f))))*val23);
    var alu27 = (val20*(1/(1.0f+exp2((val20*-1.4426950216293335f))))*val24);
    var alu28 = (val21*(1/(1.0f+exp2((val21*-1.4426950216293335f))))*val25);
    var alu29 = (val22*(1/(1.0f+exp2((val22*-1.4426950216293335f))))*val26);
    var alu30 = (val1*(1/(1.0f+exp2((val1*-1.4426950216293335f))))*val6);
    var alu31 = (val2*(1/(1.0f+exp2((val2*-1.4426950216293335f))))*val7);
    var alu32 = (val3*(1/(1.0f+exp2((val3*-1.4426950216293335f))))*val8);
    var alu33 = (val4*(1/(1.0f+exp2((val4*-1.4426950216293335f))))*val9);
    buf0[0] = (buf0[0]+(alu19*val47)+(alu18*val32)+(alu20*val37)+(alu21*val42));
    buf0[1] = (buf0[1]+(alu19*val43)+(alu18*val44)+(alu20*val45)+(alu21*val46));
    buf0[10] = (buf0[10]+(alu22*val33)+(alu23*val34)+(alu24*val35)+(alu25*val36));
    buf0[11] = (buf0[11]+(alu22*val38)+(alu23*val39)+(alu24*val40)+(alu25*val41));
    buf0[12] = (buf0[12]+(alu26*val47)+(alu27*val32)+(alu28*val37)+(alu29*val42));
    buf0[13] = (buf0[13]+(alu26*val43)+(alu27*val44)+(alu28*val45)+(alu29*val46));
    buf0[14] = (buf0[14]+(alu26*val33)+(alu27*val34)+(alu28*val35)+(alu29*val36));
    buf0[15] = (buf0[15]+(alu26*val38)+(alu27*val39)+(alu28*val40)+(alu29*val41));
    buf0[2] = (buf0[2]+(alu19*val33)+(alu18*val34)+(alu20*val35)+(alu21*val36));
    buf0[3] = (buf0[3]+(alu19*val38)+(alu18*val39)+(alu20*val40)+(alu21*val41));
    buf0[4] = (buf0[4]+(alu30*val47)+(alu31*val32)+(alu32*val37)+(alu33*val42));
    buf0[5] = (buf0[5]+(alu30*val43)+(alu31*val44)+(alu32*val45)+(alu33*val46));
    buf0[6] = (buf0[6]+(alu30*val33)+(alu31*val34)+(alu32*val35)+(alu33*val36));
    buf0[7] = (buf0[7]+(alu30*val38)+(alu31*val39)+(alu32*val40)+(alu33*val41));
    buf0[8] = (buf0[8]+(alu22*val47)+(alu23*val32)+(alu24*val37)+(alu25*val42));
    buf0[9] = (buf0[9]+(alu22*val43)+(alu23*val44)+(alu24*val45)+(alu25*val46));
  }
  var alu51 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu52 = (alu51+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u)));
  var val48 = data1_65536[alu52];
  var alu53 = (alu52+1);
  var val49 = data1_65536[alu53];
  var alu54 = (alu52+2);
  var val50 = data1_65536[alu54];
  var alu55 = (alu52+256);
  var val51 = data1_65536[alu55];
  var alu56 = (alu52+257);
  var val52 = data1_65536[alu56];
  var alu57 = (alu52+258);
  var val53 = data1_65536[alu57];
  var alu58 = (alu52+259);
  var val54 = data1_65536[alu58];
  var alu59 = (alu52+3);
  var val55 = data1_65536[alu59];
  var alu60 = (alu52+512);
  var val56 = data1_65536[alu60];
  var alu61 = (alu52+513);
  var val57 = data1_65536[alu61];
  var alu62 = (alu52+514);
  var val58 = data1_65536[alu62];
  var alu63 = (alu52+515);
  var val59 = data1_65536[alu63];
  var alu64 = (alu52+768);
  var val60 = data1_65536[alu64];
  var alu65 = (alu52+769);
  var val61 = data1_65536[alu65];
  var alu66 = (alu52+770);
  var val62 = data1_65536[alu66];
  var alu67 = (alu52+771);
  var val63 = data1_65536[alu67];
  var val64 = data2_1536[(alu51+1280)];
  var val65 = data2_1536[(alu51+1281)];
  var val66 = data2_1536[(alu51+1282)];
  var val67 = data2_1536[(alu51+1283)];
  var val68 = data5_256[alu51];
  var val69 = data5_256[(alu51+1)];
  var val70 = data5_256[(alu51+2)];
  var val71 = data5_256[(alu51+3)];
  data0_65536[alu52] = (val48+(val64*(buf0[0]+val68)));
  data0_65536[alu53] = (val49+(val65*(buf0[1]+val69)));
  data0_65536[alu54] = (val50+(val66*(buf0[2]+val70)));
  data0_65536[alu55] = (val51+(val64*(buf0[4]+val68)));
  data0_65536[alu56] = (val52+(val65*(buf0[5]+val69)));
  data0_65536[alu57] = (val53+(val66*(buf0[6]+val70)));
  data0_65536[alu58] = (val54+(val67*(buf0[7]+val71)));
  data0_65536[alu59] = (val55+(val67*(buf0[3]+val71)));
  data0_65536[alu60] = (val56+(val64*(buf0[8]+val68)));
  data0_65536[alu61] = (val57+(val65*(buf0[9]+val69)));
  data0_65536[alu62] = (val58+(val66*(buf0[10]+val70)));
  data0_65536[alu63] = (val59+(val67*(buf0[11]+val71)));
  data0_65536[alu64] = (val60+(val64*(buf0[12]+val68)));
  data0_65536[alu65] = (val61+(val65*(buf0[13]+val69)));
  data0_65536[alu66] = (val62+(val66*(buf0[14]+val70)));
  data0_65536[alu67] = (val63+(val67*(buf0[15]+val71)));
}`;

const r_16_8_4_4_32_dispatch_109 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_110 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var alu18 = (alu0+1);
  var val21 = data3_256[alu18];
  var alu19 = (alu0+2);
  var val22 = data3_256[alu19];
  var alu20 = (alu0+3);
  var val23 = data3_256[alu20];
  var val24 = data4_1536[alu0];
  var val25 = data4_1536[alu18];
  var val26 = data4_1536[alu19];
  var val27 = data4_1536[(alu0+256)];
  var val28 = data4_1536[(alu0+257)];
  var val29 = data4_1536[(alu0+258)];
  var val30 = data4_1536[(alu0+259)];
  var val31 = data4_1536[alu20];
  var alu21 = (1/val16);
  var alu22 = (1/val17);
  var alu23 = (1/val18);
  var alu24 = (1/val19);
  var alu25 = (1.0f+val27);
  var alu26 = (1.0f+val28);
  var alu27 = (1.0f+val29);
  var alu28 = (1.0f+val30);
  data0_65536[alu1] = ((val0*alu21*val20*alu25)+val24);
  data0_65536[alu2] = ((val1*alu21*val21*alu26)+val25);
  data0_65536[alu3] = ((val2*alu21*val22*alu27)+val26);
  data0_65536[alu4] = ((val3*alu22*val20*alu25)+val24);
  data0_65536[alu5] = ((val4*alu22*val21*alu26)+val25);
  data0_65536[alu6] = ((val5*alu22*val22*alu27)+val26);
  data0_65536[alu7] = ((val6*alu22*val23*alu28)+val31);
  data0_65536[alu8] = ((val7*alu21*val23*alu28)+val31);
  data0_65536[alu9] = ((val8*alu23*val20*alu25)+val24);
  data0_65536[alu10] = ((val9*alu23*val21*alu26)+val25);
  data0_65536[alu11] = ((val10*alu23*val22*alu27)+val26);
  data0_65536[alu12] = ((val11*alu23*val23*alu28)+val31);
  data0_65536[alu13] = ((val12*alu24*val20*alu25)+val24);
  data0_65536[alu14] = ((val13*alu24*val21*alu26)+val25);
  data0_65536[alu15] = ((val14*alu24*val22*alu27)+val26);
  data0_65536[alu16] = ((val15*alu24*val23*alu28)+val31);
}`;

const r_8_16_8_16_3_4_64_4_dispatch_111 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_196608:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu12 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast0);
    var val0 = data1_65536[alu12];
    var val1 = data1_65536[(alu12+1)];
    var val2 = data1_65536[(alu12+2)];
    var val3 = data1_65536[(alu12+256)];
    var val4 = data1_65536[(alu12+257)];
    var val5 = data1_65536[(alu12+258)];
    var val6 = data1_65536[(alu12+259)];
    var val7 = data1_65536[(alu12+3)];
    var val8 = data1_65536[(alu12+512)];
    var val9 = data1_65536[(alu12+513)];
    var val10 = data1_65536[(alu12+514)];
    var val11 = data1_65536[(alu12+515)];
    var val12 = data1_65536[(alu12+768)];
    var val13 = data1_65536[(alu12+769)];
    var val14 = data1_65536[(alu12+770)];
    var val15 = data1_65536[(alu12+771)];
    var alu13 = ((gidx0*12288)+(lidx1*768)+cast0);
    var val16 = data2_196608[(alu13+1)];
    var val17 = data2_196608[(alu13+2)];
    var val18 = data2_196608[(alu13+256)];
    var val19 = data2_196608[(alu13+257)];
    var val20 = data2_196608[(alu13+258)];
    var val21 = data2_196608[(alu13+259)];
    var val22 = data2_196608[(alu13+3)];
    var val23 = data2_196608[(alu13+512)];
    var val24 = data2_196608[(alu13+513)];
    var val25 = data2_196608[(alu13+514)];
    var val26 = data2_196608[(alu13+515)];
    var val27 = data2_196608[alu13];
    buf0[0] = (buf0[0]+(val0*val27)+(val1*val16)+(val2*val17)+(val7*val22));
    buf0[1] = (buf0[1]+(val0*val18)+(val1*val19)+(val2*val20)+(val7*val21));
    buf0[10] = (buf0[10]+(val12*val18)+(val13*val19)+(val14*val20)+(val15*val21));
    buf0[11] = (buf0[11]+(val12*val23)+(val13*val24)+(val14*val25)+(val15*val26));
    buf0[2] = (buf0[2]+(val0*val23)+(val1*val24)+(val2*val25)+(val7*val26));
    buf0[3] = (buf0[3]+(val3*val27)+(val4*val16)+(val5*val17)+(val6*val22));
    buf0[4] = (buf0[4]+(val3*val18)+(val4*val19)+(val5*val20)+(val6*val21));
    buf0[5] = (buf0[5]+(val3*val23)+(val4*val24)+(val5*val25)+(val6*val26));
    buf0[6] = (buf0[6]+(val8*val27)+(val9*val16)+(val10*val17)+(val11*val22));
    buf0[7] = (buf0[7]+(val8*val18)+(val9*val19)+(val10*val20)+(val11*val21));
    buf0[8] = (buf0[8]+(val8*val23)+(val9*val24)+(val10*val25)+(val11*val26));
    buf0[9] = (buf0[9]+(val12*val27)+(val13*val16)+(val14*val17)+(val15*val22));
  }
  var alu27 = ((gidx0*48)+(lidx1*3));
  var val28 = data3_768[(alu27+1)];
  var val29 = data3_768[(alu27+2)];
  var val30 = data3_768[alu27];
  var alu28 = (alu27+(gidx1*24576)+(lidx0*3072));
  data0_196608[(alu28+1)] = (buf0[1]+val28);
  data0_196608[(alu28+1536)] = (buf0[6]+val30);
  data0_196608[(alu28+1537)] = (buf0[7]+val28);
  data0_196608[(alu28+1538)] = (buf0[8]+val29);
  data0_196608[(alu28+2)] = (buf0[2]+val29);
  data0_196608[(alu28+2304)] = (buf0[9]+val30);
  data0_196608[(alu28+2305)] = (buf0[10]+val28);
  data0_196608[(alu28+2306)] = (buf0[11]+val29);
  data0_196608[(alu28+768)] = (buf0[3]+val30);
  data0_196608[(alu28+769)] = (buf0[4]+val28);
  data0_196608[(alu28+770)] = (buf0[5]+val29);
  data0_196608[alu28] = (buf0[0]+val30);
}`;

const r_8_16_8_4_4_4_dispatch_112 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[alu4];
    var val1 = data1_196608[(alu4+3072)];
    var val2 = data1_196608[(alu4+6144)];
    var val3 = data1_196608[(alu4+9216)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const r_8_16_8_4_4_4_dispatch_113 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[(alu4+256)];
    var val1 = data1_196608[(alu4+3328)];
    var val2 = data1_196608[(alu4+6400)];
    var val3 = data1_196608[(alu4+9472)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const E_2_32_8_16_2_4_dispatch_114 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+1)];
  var val1 = data1_196608[(alu0+32)];
  var val2 = data1_196608[(alu0+33)];
  var val3 = data1_196608[(alu0+64)];
  var val4 = data1_196608[(alu0+65)];
  var val5 = data1_196608[(alu0+96)];
  var val6 = data1_196608[(alu0+97)];
  var val7 = data1_196608[alu0];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val13);
  var alu10 = (val7*alu5*val12);
  var alu11 = (val1*alu6*val12);
  var alu12 = (val2*alu6*val13);
  var alu13 = (val3*alu7*val12);
  var alu14 = (val4*alu7*val13);
  var alu15 = (val5*alu8*val12);
  var alu16 = (val6*alu8*val13);
  data0_65536[alu4] = ((alu10*val14)-(alu9*val16));
  data0_65536[(alu4+1)] = ((alu9*val15)+(alu10*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const E_2_32_8_16_2_4_dispatch_115 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+256)];
  var val1 = data1_196608[(alu0+257)];
  var val2 = data1_196608[(alu0+288)];
  var val3 = data1_196608[(alu0+289)];
  var val4 = data1_196608[(alu0+320)];
  var val5 = data1_196608[(alu0+321)];
  var val6 = data1_196608[(alu0+352)];
  var val7 = data1_196608[(alu0+353)];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val12);
  var alu10 = (val1*alu5*val13);
  var alu11 = (val2*alu6*val12);
  var alu12 = (val3*alu6*val13);
  var alu13 = (val4*alu7*val12);
  var alu14 = (val5*alu7*val13);
  var alu15 = (val6*alu8*val12);
  var alu16 = (val7*alu8*val13);
  data0_65536[alu4] = ((alu9*val14)-(alu10*val16));
  data0_65536[(alu4+1)] = ((alu10*val15)+(alu9*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const r_8_8_4_8_16_4_4_32_dispatch_116 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_524288:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
  var cast2 = bitcast<i32>((cast1<<13u));
  var cast3 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<10u))+bitcast<i32>((cast3<<7u))+cast2);
  var val0 = data1_65536[alu0];
  var val1 = data1_65536[(alu0+1)];
  var val2 = data1_65536[(alu0+10)];
  var val3 = data1_65536[(alu0+100)];
  var val4 = data1_65536[(alu0+101)];
  var val5 = data1_65536[(alu0+102)];
  var val6 = data1_65536[(alu0+103)];
  var val7 = data1_65536[(alu0+104)];
  var val8 = data1_65536[(alu0+105)];
  var val9 = data1_65536[(alu0+106)];
  var val10 = data1_65536[(alu0+107)];
  var val11 = data1_65536[(alu0+108)];
  var val12 = data1_65536[(alu0+109)];
  var val13 = data1_65536[(alu0+11)];
  var val14 = data1_65536[(alu0+110)];
  var val15 = data1_65536[(alu0+111)];
  var val16 = data1_65536[(alu0+112)];
  var val17 = data1_65536[(alu0+113)];
  var val18 = data1_65536[(alu0+114)];
  var val19 = data1_65536[(alu0+115)];
  var val20 = data1_65536[(alu0+116)];
  var val21 = data1_65536[(alu0+117)];
  var val22 = data1_65536[(alu0+118)];
  var val23 = data1_65536[(alu0+119)];
  var val24 = data1_65536[(alu0+12)];
  var val25 = data1_65536[(alu0+120)];
  var val26 = data1_65536[(alu0+121)];
  var val27 = data1_65536[(alu0+122)];
  var val28 = data1_65536[(alu0+123)];
  var val29 = data1_65536[(alu0+124)];
  var val30 = data1_65536[(alu0+125)];
  var val31 = data1_65536[(alu0+126)];
  var val32 = data1_65536[(alu0+127)];
  var val33 = data1_65536[(alu0+13)];
  var val34 = data1_65536[(alu0+14)];
  var val35 = data1_65536[(alu0+15)];
  var val36 = data1_65536[(alu0+16)];
  var val37 = data1_65536[(alu0+17)];
  var val38 = data1_65536[(alu0+18)];
  var val39 = data1_65536[(alu0+19)];
  var val40 = data1_65536[(alu0+2)];
  var val41 = data1_65536[(alu0+20)];
  var val42 = data1_65536[(alu0+21)];
  var val43 = data1_65536[(alu0+22)];
  var val44 = data1_65536[(alu0+23)];
  var val45 = data1_65536[(alu0+24)];
  var val46 = data1_65536[(alu0+25)];
  var val47 = data1_65536[(alu0+26)];
  var val48 = data1_65536[(alu0+27)];
  var val49 = data1_65536[(alu0+28)];
  var val50 = data1_65536[(alu0+29)];
  var val51 = data1_65536[(alu0+3)];
  var val52 = data1_65536[(alu0+30)];
  var val53 = data1_65536[(alu0+31)];
  var val54 = data1_65536[(alu0+32)];
  var val55 = data1_65536[(alu0+33)];
  var val56 = data1_65536[(alu0+34)];
  var val57 = data1_65536[(alu0+35)];
  var val58 = data1_65536[(alu0+36)];
  var val59 = data1_65536[(alu0+37)];
  var val60 = data1_65536[(alu0+38)];
  var val61 = data1_65536[(alu0+39)];
  var val62 = data1_65536[(alu0+4)];
  var val63 = data1_65536[(alu0+40)];
  var val64 = data1_65536[(alu0+41)];
  var val65 = data1_65536[(alu0+42)];
  var val66 = data1_65536[(alu0+43)];
  var val67 = data1_65536[(alu0+44)];
  var val68 = data1_65536[(alu0+45)];
  var val69 = data1_65536[(alu0+46)];
  var val70 = data1_65536[(alu0+47)];
  var val71 = data1_65536[(alu0+48)];
  var val72 = data1_65536[(alu0+49)];
  var val73 = data1_65536[(alu0+5)];
  var val74 = data1_65536[(alu0+50)];
  var val75 = data1_65536[(alu0+51)];
  var val76 = data1_65536[(alu0+52)];
  var val77 = data1_65536[(alu0+53)];
  var val78 = data1_65536[(alu0+54)];
  var val79 = data1_65536[(alu0+55)];
  var val80 = data1_65536[(alu0+56)];
  var val81 = data1_65536[(alu0+57)];
  var val82 = data1_65536[(alu0+58)];
  var val83 = data1_65536[(alu0+59)];
  var val84 = data1_65536[(alu0+6)];
  var val85 = data1_65536[(alu0+60)];
  var val86 = data1_65536[(alu0+61)];
  var val87 = data1_65536[(alu0+62)];
  var val88 = data1_65536[(alu0+63)];
  var val89 = data1_65536[(alu0+64)];
  var val90 = data1_65536[(alu0+65)];
  var val91 = data1_65536[(alu0+66)];
  var val92 = data1_65536[(alu0+67)];
  var val93 = data1_65536[(alu0+68)];
  var val94 = data1_65536[(alu0+69)];
  var val95 = data1_65536[(alu0+7)];
  var val96 = data1_65536[(alu0+70)];
  var val97 = data1_65536[(alu0+71)];
  var val98 = data1_65536[(alu0+72)];
  var val99 = data1_65536[(alu0+73)];
  var val100 = data1_65536[(alu0+74)];
  var val101 = data1_65536[(alu0+75)];
  var val102 = data1_65536[(alu0+76)];
  var val103 = data1_65536[(alu0+77)];
  var val104 = data1_65536[(alu0+78)];
  var val105 = data1_65536[(alu0+79)];
  var val106 = data1_65536[(alu0+8)];
  var val107 = data1_65536[(alu0+80)];
  var val108 = data1_65536[(alu0+81)];
  var val109 = data1_65536[(alu0+82)];
  var val110 = data1_65536[(alu0+83)];
  var val111 = data1_65536[(alu0+84)];
  var val112 = data1_65536[(alu0+85)];
  var val113 = data1_65536[(alu0+86)];
  var val114 = data1_65536[(alu0+87)];
  var val115 = data1_65536[(alu0+88)];
  var val116 = data1_65536[(alu0+89)];
  var val117 = data1_65536[(alu0+9)];
  var val118 = data1_65536[(alu0+90)];
  var val119 = data1_65536[(alu0+91)];
  var val120 = data1_65536[(alu0+92)];
  var val121 = data1_65536[(alu0+93)];
  var val122 = data1_65536[(alu0+94)];
  var val123 = data1_65536[(alu0+95)];
  var val124 = data1_65536[(alu0+96)];
  var val125 = data1_65536[(alu0+97)];
  var val126 = data1_65536[(alu0+98)];
  var val127 = data1_65536[(alu0+99)];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast4 = bitcast<u32>(gidx0);
  var cast5 = bitcast<u32>(lidx1);
  var alu1 = (bitcast<i32>((cast4<<11u))+bitcast<i32>((cast5<<7u))+cast2);
  var val128 = data2_65536[alu1];
  var val129 = data2_65536[(alu1+1)];
  var val130 = data2_65536[(alu1+10)];
  var val131 = data2_65536[(alu1+100)];
  var val132 = data2_65536[(alu1+101)];
  var val133 = data2_65536[(alu1+102)];
  var val134 = data2_65536[(alu1+103)];
  var val135 = data2_65536[(alu1+104)];
  var val136 = data2_65536[(alu1+105)];
  var val137 = data2_65536[(alu1+106)];
  var val138 = data2_65536[(alu1+107)];
  var val139 = data2_65536[(alu1+108)];
  var val140 = data2_65536[(alu1+109)];
  var val141 = data2_65536[(alu1+11)];
  var val142 = data2_65536[(alu1+110)];
  var val143 = data2_65536[(alu1+111)];
  var val144 = data2_65536[(alu1+112)];
  var val145 = data2_65536[(alu1+113)];
  var val146 = data2_65536[(alu1+114)];
  var val147 = data2_65536[(alu1+115)];
  var val148 = data2_65536[(alu1+116)];
  var val149 = data2_65536[(alu1+117)];
  var val150 = data2_65536[(alu1+118)];
  var val151 = data2_65536[(alu1+119)];
  var val152 = data2_65536[(alu1+12)];
  var val153 = data2_65536[(alu1+120)];
  var val154 = data2_65536[(alu1+121)];
  var val155 = data2_65536[(alu1+122)];
  var val156 = data2_65536[(alu1+123)];
  var val157 = data2_65536[(alu1+124)];
  var val158 = data2_65536[(alu1+125)];
  var val159 = data2_65536[(alu1+126)];
  var val160 = data2_65536[(alu1+127)];
  var val161 = data2_65536[(alu1+13)];
  var val162 = data2_65536[(alu1+14)];
  var val163 = data2_65536[(alu1+15)];
  var val164 = data2_65536[(alu1+16)];
  var val165 = data2_65536[(alu1+17)];
  var val166 = data2_65536[(alu1+18)];
  var val167 = data2_65536[(alu1+19)];
  var val168 = data2_65536[(alu1+2)];
  var val169 = data2_65536[(alu1+20)];
  var val170 = data2_65536[(alu1+21)];
  var val171 = data2_65536[(alu1+22)];
  var val172 = data2_65536[(alu1+23)];
  var val173 = data2_65536[(alu1+24)];
  var val174 = data2_65536[(alu1+25)];
  var val175 = data2_65536[(alu1+26)];
  var val176 = data2_65536[(alu1+27)];
  var val177 = data2_65536[(alu1+28)];
  var val178 = data2_65536[(alu1+29)];
  var val179 = data2_65536[(alu1+3)];
  var val180 = data2_65536[(alu1+30)];
  var val181 = data2_65536[(alu1+31)];
  var val182 = data2_65536[(alu1+32)];
  var val183 = data2_65536[(alu1+33)];
  var val184 = data2_65536[(alu1+34)];
  var val185 = data2_65536[(alu1+35)];
  var val186 = data2_65536[(alu1+36)];
  var val187 = data2_65536[(alu1+37)];
  var val188 = data2_65536[(alu1+38)];
  var val189 = data2_65536[(alu1+39)];
  var val190 = data2_65536[(alu1+4)];
  var val191 = data2_65536[(alu1+40)];
  var val192 = data2_65536[(alu1+41)];
  var val193 = data2_65536[(alu1+42)];
  var val194 = data2_65536[(alu1+43)];
  var val195 = data2_65536[(alu1+44)];
  var val196 = data2_65536[(alu1+45)];
  var val197 = data2_65536[(alu1+46)];
  var val198 = data2_65536[(alu1+47)];
  var val199 = data2_65536[(alu1+48)];
  var val200 = data2_65536[(alu1+49)];
  var val201 = data2_65536[(alu1+5)];
  var val202 = data2_65536[(alu1+50)];
  var val203 = data2_65536[(alu1+51)];
  var val204 = data2_65536[(alu1+52)];
  var val205 = data2_65536[(alu1+53)];
  var val206 = data2_65536[(alu1+54)];
  var val207 = data2_65536[(alu1+55)];
  var val208 = data2_65536[(alu1+56)];
  var val209 = data2_65536[(alu1+57)];
  var val210 = data2_65536[(alu1+58)];
  var val211 = data2_65536[(alu1+59)];
  var val212 = data2_65536[(alu1+6)];
  var val213 = data2_65536[(alu1+60)];
  var val214 = data2_65536[(alu1+61)];
  var val215 = data2_65536[(alu1+62)];
  var val216 = data2_65536[(alu1+63)];
  var val217 = data2_65536[(alu1+64)];
  var val218 = data2_65536[(alu1+65)];
  var val219 = data2_65536[(alu1+66)];
  var val220 = data2_65536[(alu1+67)];
  var val221 = data2_65536[(alu1+68)];
  var val222 = data2_65536[(alu1+69)];
  var val223 = data2_65536[(alu1+7)];
  var val224 = data2_65536[(alu1+70)];
  var val225 = data2_65536[(alu1+71)];
  var val226 = data2_65536[(alu1+72)];
  var val227 = data2_65536[(alu1+73)];
  var val228 = data2_65536[(alu1+74)];
  var val229 = data2_65536[(alu1+75)];
  var val230 = data2_65536[(alu1+76)];
  var val231 = data2_65536[(alu1+77)];
  var val232 = data2_65536[(alu1+78)];
  var val233 = data2_65536[(alu1+79)];
  var val234 = data2_65536[(alu1+8)];
  var val235 = data2_65536[(alu1+80)];
  var val236 = data2_65536[(alu1+81)];
  var val237 = data2_65536[(alu1+82)];
  var val238 = data2_65536[(alu1+83)];
  var val239 = data2_65536[(alu1+84)];
  var val240 = data2_65536[(alu1+85)];
  var val241 = data2_65536[(alu1+86)];
  var val242 = data2_65536[(alu1+87)];
  var val243 = data2_65536[(alu1+88)];
  var val244 = data2_65536[(alu1+89)];
  var val245 = data2_65536[(alu1+9)];
  var val246 = data2_65536[(alu1+90)];
  var val247 = data2_65536[(alu1+91)];
  var val248 = data2_65536[(alu1+92)];
  var val249 = data2_65536[(alu1+93)];
  var val250 = data2_65536[(alu1+94)];
  var val251 = data2_65536[(alu1+95)];
  var val252 = data2_65536[(alu1+96)];
  var val253 = data2_65536[(alu1+97)];
  var val254 = data2_65536[(alu1+98)];
  var val255 = data2_65536[(alu1+99)];
  var alu2 = (bitcast<i32>((cast4<<6u))+bitcast<i32>((cast5<<2u))+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast1<<16u)));
  data0_524288[alu2] = (((val0*val128)+(val1*val129)+(val40*val168)+(val51*val179)+(val62*val190)+(val73*val201)+(val84*val212)+(val95*val223)+(val106*val234)+(val117*val245)+(val2*val130)+(val13*val141)+(val24*val152)+(val33*val161)+(val34*val162)+(val35*val163)+(val36*val164)+(val37*val165)+(val38*val166)+(val39*val167)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val173)+(val46*val174)+(val47*val175)+(val48*val176)+(val49*val177)+(val50*val178)+(val52*val180)+(val53*val181))*0.1767766922712326f);
  data0_524288[(alu2+1)] = (((val0*val182)+(val1*val183)+(val40*val184)+(val51*val185)+(val62*val186)+(val73*val187)+(val84*val188)+(val95*val189)+(val106*val191)+(val117*val192)+(val2*val193)+(val13*val194)+(val24*val195)+(val33*val196)+(val34*val197)+(val35*val198)+(val36*val199)+(val37*val200)+(val38*val202)+(val39*val203)+(val41*val204)+(val42*val205)+(val43*val206)+(val44*val207)+(val45*val208)+(val46*val209)+(val47*val210)+(val48*val211)+(val49*val213)+(val50*val214)+(val52*val215)+(val53*val216))*0.1767766922712326f);
  data0_524288[(alu2+2)] = (((val0*val217)+(val1*val218)+(val40*val219)+(val51*val220)+(val62*val221)+(val73*val222)+(val84*val224)+(val95*val225)+(val106*val226)+(val117*val227)+(val2*val228)+(val13*val229)+(val24*val230)+(val33*val231)+(val34*val232)+(val35*val233)+(val36*val235)+(val37*val236)+(val38*val237)+(val39*val238)+(val41*val239)+(val42*val240)+(val43*val241)+(val44*val242)+(val45*val243)+(val46*val244)+(val47*val246)+(val48*val247)+(val49*val248)+(val50*val249)+(val52*val250)+(val53*val251))*0.1767766922712326f);
  data0_524288[(alu2+256)] = (((val54*val128)+(val55*val129)+(val56*val168)+(val57*val179)+(val58*val190)+(val59*val201)+(val60*val212)+(val61*val223)+(val63*val234)+(val64*val245)+(val65*val130)+(val66*val141)+(val67*val152)+(val68*val161)+(val69*val162)+(val70*val163)+(val71*val164)+(val72*val165)+(val74*val166)+(val75*val167)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val173)+(val81*val174)+(val82*val175)+(val83*val176)+(val85*val177)+(val86*val178)+(val87*val180)+(val88*val181))*0.1767766922712326f);
  data0_524288[(alu2+257)] = (((val54*val182)+(val55*val183)+(val56*val184)+(val57*val185)+(val58*val186)+(val59*val187)+(val60*val188)+(val61*val189)+(val63*val191)+(val64*val192)+(val65*val193)+(val66*val194)+(val67*val195)+(val68*val196)+(val69*val197)+(val70*val198)+(val71*val199)+(val72*val200)+(val74*val202)+(val75*val203)+(val76*val204)+(val77*val205)+(val78*val206)+(val79*val207)+(val80*val208)+(val81*val209)+(val82*val210)+(val83*val211)+(val85*val213)+(val86*val214)+(val87*val215)+(val88*val216))*0.1767766922712326f);
  data0_524288[(alu2+258)] = (((val54*val217)+(val55*val218)+(val56*val219)+(val57*val220)+(val58*val221)+(val59*val222)+(val60*val224)+(val61*val225)+(val63*val226)+(val64*val227)+(val65*val228)+(val66*val229)+(val67*val230)+(val68*val231)+(val69*val232)+(val70*val233)+(val71*val235)+(val72*val236)+(val74*val237)+(val75*val238)+(val76*val239)+(val77*val240)+(val78*val241)+(val79*val242)+(val80*val243)+(val81*val244)+(val82*val246)+(val83*val247)+(val85*val248)+(val86*val249)+(val87*val250)+(val88*val251))*0.1767766922712326f);
  data0_524288[(alu2+259)] = (((val54*val252)+(val55*val253)+(val56*val254)+(val57*val255)+(val58*val131)+(val59*val132)+(val60*val133)+(val61*val134)+(val63*val135)+(val64*val136)+(val65*val137)+(val66*val138)+(val67*val139)+(val68*val140)+(val69*val142)+(val70*val143)+(val71*val144)+(val72*val145)+(val74*val146)+(val75*val147)+(val76*val148)+(val77*val149)+(val78*val150)+(val79*val151)+(val80*val153)+(val81*val154)+(val82*val155)+(val83*val156)+(val85*val157)+(val86*val158)+(val87*val159)+(val88*val160))*0.1767766922712326f);
  data0_524288[(alu2+3)] = (((val0*val252)+(val1*val253)+(val40*val254)+(val51*val255)+(val62*val131)+(val73*val132)+(val84*val133)+(val95*val134)+(val106*val135)+(val117*val136)+(val2*val137)+(val13*val138)+(val24*val139)+(val33*val140)+(val34*val142)+(val35*val143)+(val36*val144)+(val37*val145)+(val38*val146)+(val39*val147)+(val41*val148)+(val42*val149)+(val43*val150)+(val44*val151)+(val45*val153)+(val46*val154)+(val47*val155)+(val48*val156)+(val49*val157)+(val50*val158)+(val52*val159)+(val53*val160))*0.1767766922712326f);
  data0_524288[(alu2+512)] = (((val89*val128)+(val90*val129)+(val91*val168)+(val92*val179)+(val93*val190)+(val94*val201)+(val96*val212)+(val97*val223)+(val98*val234)+(val99*val245)+(val100*val130)+(val101*val141)+(val102*val152)+(val103*val161)+(val104*val162)+(val105*val163)+(val107*val164)+(val108*val165)+(val109*val166)+(val110*val167)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val173)+(val116*val174)+(val118*val175)+(val119*val176)+(val120*val177)+(val121*val178)+(val122*val180)+(val123*val181))*0.1767766922712326f);
  data0_524288[(alu2+513)] = (((val89*val182)+(val90*val183)+(val91*val184)+(val92*val185)+(val93*val186)+(val94*val187)+(val96*val188)+(val97*val189)+(val98*val191)+(val99*val192)+(val100*val193)+(val101*val194)+(val102*val195)+(val103*val196)+(val104*val197)+(val105*val198)+(val107*val199)+(val108*val200)+(val109*val202)+(val110*val203)+(val111*val204)+(val112*val205)+(val113*val206)+(val114*val207)+(val115*val208)+(val116*val209)+(val118*val210)+(val119*val211)+(val120*val213)+(val121*val214)+(val122*val215)+(val123*val216))*0.1767766922712326f);
  data0_524288[(alu2+514)] = (((val89*val217)+(val90*val218)+(val91*val219)+(val92*val220)+(val93*val221)+(val94*val222)+(val96*val224)+(val97*val225)+(val98*val226)+(val99*val227)+(val100*val228)+(val101*val229)+(val102*val230)+(val103*val231)+(val104*val232)+(val105*val233)+(val107*val235)+(val108*val236)+(val109*val237)+(val110*val238)+(val111*val239)+(val112*val240)+(val113*val241)+(val114*val242)+(val115*val243)+(val116*val244)+(val118*val246)+(val119*val247)+(val120*val248)+(val121*val249)+(val122*val250)+(val123*val251))*0.1767766922712326f);
  data0_524288[(alu2+515)] = (((val89*val252)+(val90*val253)+(val91*val254)+(val92*val255)+(val93*val131)+(val94*val132)+(val96*val133)+(val97*val134)+(val98*val135)+(val99*val136)+(val100*val137)+(val101*val138)+(val102*val139)+(val103*val140)+(val104*val142)+(val105*val143)+(val107*val144)+(val108*val145)+(val109*val146)+(val110*val147)+(val111*val148)+(val112*val149)+(val113*val150)+(val114*val151)+(val115*val153)+(val116*val154)+(val118*val155)+(val119*val156)+(val120*val157)+(val121*val158)+(val122*val159)+(val123*val160))*0.1767766922712326f);
  data0_524288[(alu2+768)] = (((val124*val128)+(val125*val129)+(val126*val168)+(val127*val179)+(val3*val190)+(val4*val201)+(val5*val212)+(val6*val223)+(val7*val234)+(val8*val245)+(val9*val130)+(val10*val141)+(val11*val152)+(val12*val161)+(val14*val162)+(val15*val163)+(val16*val164)+(val17*val165)+(val18*val166)+(val19*val167)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val173)+(val26*val174)+(val27*val175)+(val28*val176)+(val29*val177)+(val30*val178)+(val31*val180)+(val32*val181))*0.1767766922712326f);
  data0_524288[(alu2+769)] = (((val124*val182)+(val125*val183)+(val126*val184)+(val127*val185)+(val3*val186)+(val4*val187)+(val5*val188)+(val6*val189)+(val7*val191)+(val8*val192)+(val9*val193)+(val10*val194)+(val11*val195)+(val12*val196)+(val14*val197)+(val15*val198)+(val16*val199)+(val17*val200)+(val18*val202)+(val19*val203)+(val20*val204)+(val21*val205)+(val22*val206)+(val23*val207)+(val25*val208)+(val26*val209)+(val27*val210)+(val28*val211)+(val29*val213)+(val30*val214)+(val31*val215)+(val32*val216))*0.1767766922712326f);
  data0_524288[(alu2+770)] = (((val124*val217)+(val125*val218)+(val126*val219)+(val127*val220)+(val3*val221)+(val4*val222)+(val5*val224)+(val6*val225)+(val7*val226)+(val8*val227)+(val9*val228)+(val10*val229)+(val11*val230)+(val12*val231)+(val14*val232)+(val15*val233)+(val16*val235)+(val17*val236)+(val18*val237)+(val19*val238)+(val20*val239)+(val21*val240)+(val22*val241)+(val23*val242)+(val25*val243)+(val26*val244)+(val27*val246)+(val28*val247)+(val29*val248)+(val30*val249)+(val31*val250)+(val32*val251))*0.1767766922712326f);
  data0_524288[(alu2+771)] = (((val124*val252)+(val125*val253)+(val126*val254)+(val127*val255)+(val3*val131)+(val4*val132)+(val5*val133)+(val6*val134)+(val7*val135)+(val8*val136)+(val9*val137)+(val10*val138)+(val11*val139)+(val12*val140)+(val14*val142)+(val15*val143)+(val16*val144)+(val17*val145)+(val18*val146)+(val19*val147)+(val20*val148)+(val21*val149)+(val22*val150)+(val23*val151)+(val25*val153)+(val26*val154)+(val27*val155)+(val28*val156)+(val29*val157)+(val30*val158)+(val31*val159)+(val32*val160))*0.1767766922712326f);
}`;

const r_2048_16_16_dispatch_117 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = (f32(-INFINITY));
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val0 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    var val1 = buf0[0];
    var alu1 = select(val1,val0,(val1<val0));
    buf0[0] = alu1;
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = (f32(-INFINITY));
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf1[0];
    var val3 = buf2[Lidx102];
    var alu7 = select(val2,val3,(val2<val3));
    buf1[0] = alu7;
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_2048_16_16_dispatch_118 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var val0 = data2_2048[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val1 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    buf0[0] = (buf0[0]+exp2(((val1-val0)*1.4426950216293335f)));
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf2[Lidx102];
    buf1[0] = (buf1[0]+val2);
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_8_4_16_8_4_4_64_4_dispatch_119 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2048:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_196608:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u))+bitcast<i32>((cast1<<8u)));
  var val0 = data2_2048[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_2048[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_2048[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_2048[alu3];
  var lidx1 = i32(lindex.y); /* 8 */
  var cast3 = bitcast<i32>((bitcast<u32>(lidx1)<<2u));
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var alu20 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast2<<10u))+bitcast<i32>((bitcast<u32>(Lidx0)<<2u))+bitcast<i32>((cast1<<16u)));
    var val4 = data1_524288[alu20];
    var val5 = data1_524288[(alu20+1)];
    var val6 = data1_524288[(alu20+2)];
    var val7 = data1_524288[(alu20+256)];
    var val8 = data1_524288[(alu20+257)];
    var val9 = data1_524288[(alu20+258)];
    var val10 = data1_524288[(alu20+259)];
    var val11 = data1_524288[(alu20+3)];
    var val12 = data1_524288[(alu20+512)];
    var val13 = data1_524288[(alu20+513)];
    var val14 = data1_524288[(alu20+514)];
    var val15 = data1_524288[(alu20+515)];
    var val16 = data1_524288[(alu20+768)];
    var val17 = data1_524288[(alu20+769)];
    var val18 = data1_524288[(alu20+770)];
    var val19 = data1_524288[(alu20+771)];
    var alu21 = (bitcast<i32>((cast1<<5u))+cast3+(Lidx0*3072));
    var val20 = data4_196608[(alu21+1280)];
    var val21 = data4_196608[(alu21+1281)];
    var val22 = data4_196608[(alu21+1282)];
    var val23 = data4_196608[(alu21+1283)];
    var val24 = data4_196608[(alu21+2048)];
    var val25 = data4_196608[(alu21+2049)];
    var val26 = data4_196608[(alu21+2050)];
    var val27 = data4_196608[(alu21+2051)];
    var val28 = data4_196608[(alu21+2816)];
    var val29 = data4_196608[(alu21+2817)];
    var val30 = data4_196608[(alu21+2818)];
    var val31 = data4_196608[(alu21+2819)];
    var val32 = data4_196608[(alu21+512)];
    var val33 = data4_196608[(alu21+513)];
    var val34 = data4_196608[(alu21+514)];
    var val35 = data4_196608[(alu21+515)];
    var alu22 = exp2(((val4-val0)*1.4426950216293335f));
    var alu23 = exp2(((val5-val0)*1.4426950216293335f));
    var alu24 = exp2(((val6-val0)*1.4426950216293335f));
    var alu25 = exp2(((val7-val1)*1.4426950216293335f));
    var alu26 = exp2(((val8-val1)*1.4426950216293335f));
    var alu27 = exp2(((val9-val1)*1.4426950216293335f));
    var alu28 = exp2(((val10-val1)*1.4426950216293335f));
    var alu29 = exp2(((val11-val0)*1.4426950216293335f));
    var alu30 = exp2(((val12-val2)*1.4426950216293335f));
    var alu31 = exp2(((val13-val2)*1.4426950216293335f));
    var alu32 = exp2(((val14-val2)*1.4426950216293335f));
    var alu33 = exp2(((val15-val2)*1.4426950216293335f));
    var alu34 = exp2(((val16-val3)*1.4426950216293335f));
    var alu35 = exp2(((val17-val3)*1.4426950216293335f));
    var alu36 = exp2(((val18-val3)*1.4426950216293335f));
    var alu37 = exp2(((val19-val3)*1.4426950216293335f));
    buf0[0] = (buf0[0]+(alu22*val32)+(alu23*val20)+(alu24*val24)+(alu29*val28));
    buf0[1] = (buf0[1]+(alu22*val33)+(alu23*val21)+(alu24*val25)+(alu29*val29));
    buf0[10] = (buf0[10]+(alu30*val34)+(alu31*val22)+(alu32*val26)+(alu33*val30));
    buf0[11] = (buf0[11]+(alu30*val35)+(alu31*val23)+(alu32*val27)+(alu33*val31));
    buf0[12] = (buf0[12]+(alu34*val32)+(alu35*val20)+(alu36*val24)+(alu37*val28));
    buf0[13] = (buf0[13]+(alu34*val33)+(alu35*val21)+(alu36*val25)+(alu37*val29));
    buf0[14] = (buf0[14]+(alu34*val34)+(alu35*val22)+(alu36*val26)+(alu37*val30));
    buf0[15] = (buf0[15]+(alu34*val35)+(alu35*val23)+(alu36*val27)+(alu37*val31));
    buf0[2] = (buf0[2]+(alu22*val34)+(alu23*val22)+(alu24*val26)+(alu29*val30));
    buf0[3] = (buf0[3]+(alu22*val35)+(alu23*val23)+(alu24*val27)+(alu29*val31));
    buf0[4] = (buf0[4]+(alu25*val32)+(alu26*val20)+(alu27*val24)+(alu28*val28));
    buf0[5] = (buf0[5]+(alu25*val33)+(alu26*val21)+(alu27*val25)+(alu28*val29));
    buf0[6] = (buf0[6]+(alu25*val34)+(alu26*val22)+(alu27*val26)+(alu28*val30));
    buf0[7] = (buf0[7]+(alu25*val35)+(alu26*val23)+(alu27*val27)+(alu28*val31));
    buf0[8] = (buf0[8]+(alu30*val32)+(alu31*val20)+(alu32*val24)+(alu33*val28));
    buf0[9] = (buf0[9]+(alu30*val33)+(alu31*val21)+(alu32*val25)+(alu33*val29));
  }
  var val36 = data3_2048[alu0];
  var val37 = data3_2048[alu1];
  var val38 = data3_2048[alu2];
  var val39 = data3_2048[alu3];
  var alu55 = (bitcast<i32>((cast0<<11u))+bitcast<i32>((cast2<<7u))+cast3+bitcast<i32>((cast1<<13u)));
  var alu56 = (1/val36);
  var alu57 = (1/val37);
  var alu58 = (1/val38);
  var alu59 = (1/val39);
  data0_65536[alu55] = (buf0[0]*alu56);
  data0_65536[(alu55+1)] = (buf0[1]*alu56);
  data0_65536[(alu55+2)] = (buf0[2]*alu56);
  data0_65536[(alu55+3)] = (buf0[3]*alu56);
  data0_65536[(alu55+32)] = (buf0[4]*alu57);
  data0_65536[(alu55+33)] = (buf0[5]*alu57);
  data0_65536[(alu55+34)] = (buf0[6]*alu57);
  data0_65536[(alu55+35)] = (buf0[7]*alu57);
  data0_65536[(alu55+64)] = (buf0[8]*alu58);
  data0_65536[(alu55+65)] = (buf0[9]*alu58);
  data0_65536[(alu55+66)] = (buf0[10]*alu58);
  data0_65536[(alu55+67)] = (buf0[11]*alu58);
  data0_65536[(alu55+96)] = (buf0[12]*alu59);
  data0_65536[(alu55+97)] = (buf0[13]*alu59);
  data0_65536[(alu55+98)] = (buf0[14]*alu59);
  data0_65536[(alu55+99)] = (buf0[15]*alu59);
}`;

const r_8_4_8_16_4_4_8_32_dispatch_120 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_65536:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0_0 = 0; Lidx0_0 < 8; Lidx0_0++) {
    var cast4 = bitcast<u32>(Lidx0_0);
    var alu16 = (bitcast<i32>((cast1<<10u))+bitcast<i32>((cast2<<7u))+bitcast<i32>((cast4<<13u)));
    var val0 = data3_65536[alu16];
    var val1 = data3_65536[(alu16+1)];
    var val2 = data3_65536[(alu16+10)];
    var val3 = data3_65536[(alu16+100)];
    var val4 = data3_65536[(alu16+101)];
    var val5 = data3_65536[(alu16+102)];
    var val6 = data3_65536[(alu16+103)];
    var val7 = data3_65536[(alu16+104)];
    var val8 = data3_65536[(alu16+105)];
    var val9 = data3_65536[(alu16+106)];
    var val10 = data3_65536[(alu16+107)];
    var val11 = data3_65536[(alu16+108)];
    var val12 = data3_65536[(alu16+109)];
    var val13 = data3_65536[(alu16+11)];
    var val14 = data3_65536[(alu16+110)];
    var val15 = data3_65536[(alu16+111)];
    var val16 = data3_65536[(alu16+112)];
    var val17 = data3_65536[(alu16+113)];
    var val18 = data3_65536[(alu16+114)];
    var val19 = data3_65536[(alu16+115)];
    var val20 = data3_65536[(alu16+116)];
    var val21 = data3_65536[(alu16+117)];
    var val22 = data3_65536[(alu16+118)];
    var val23 = data3_65536[(alu16+119)];
    var val24 = data3_65536[(alu16+12)];
    var val25 = data3_65536[(alu16+120)];
    var val26 = data3_65536[(alu16+121)];
    var val27 = data3_65536[(alu16+122)];
    var val28 = data3_65536[(alu16+123)];
    var val29 = data3_65536[(alu16+124)];
    var val30 = data3_65536[(alu16+125)];
    var val31 = data3_65536[(alu16+126)];
    var val32 = data3_65536[(alu16+127)];
    var val33 = data3_65536[(alu16+13)];
    var val34 = data3_65536[(alu16+14)];
    var val35 = data3_65536[(alu16+15)];
    var val36 = data3_65536[(alu16+16)];
    var val37 = data3_65536[(alu16+17)];
    var val38 = data3_65536[(alu16+18)];
    var val39 = data3_65536[(alu16+19)];
    var val40 = data3_65536[(alu16+2)];
    var val41 = data3_65536[(alu16+20)];
    var val42 = data3_65536[(alu16+21)];
    var val43 = data3_65536[(alu16+22)];
    var val44 = data3_65536[(alu16+23)];
    var val45 = data3_65536[(alu16+24)];
    var val46 = data3_65536[(alu16+25)];
    var val47 = data3_65536[(alu16+26)];
    var val48 = data3_65536[(alu16+27)];
    var val49 = data3_65536[(alu16+28)];
    var val50 = data3_65536[(alu16+29)];
    var val51 = data3_65536[(alu16+3)];
    var val52 = data3_65536[(alu16+30)];
    var val53 = data3_65536[(alu16+31)];
    var val54 = data3_65536[(alu16+32)];
    var val55 = data3_65536[(alu16+33)];
    var val56 = data3_65536[(alu16+34)];
    var val57 = data3_65536[(alu16+35)];
    var val58 = data3_65536[(alu16+36)];
    var val59 = data3_65536[(alu16+37)];
    var val60 = data3_65536[(alu16+38)];
    var val61 = data3_65536[(alu16+39)];
    var val62 = data3_65536[(alu16+4)];
    var val63 = data3_65536[(alu16+40)];
    var val64 = data3_65536[(alu16+41)];
    var val65 = data3_65536[(alu16+42)];
    var val66 = data3_65536[(alu16+43)];
    var val67 = data3_65536[(alu16+44)];
    var val68 = data3_65536[(alu16+45)];
    var val69 = data3_65536[(alu16+46)];
    var val70 = data3_65536[(alu16+47)];
    var val71 = data3_65536[(alu16+48)];
    var val72 = data3_65536[(alu16+49)];
    var val73 = data3_65536[(alu16+5)];
    var val74 = data3_65536[(alu16+50)];
    var val75 = data3_65536[(alu16+51)];
    var val76 = data3_65536[(alu16+52)];
    var val77 = data3_65536[(alu16+53)];
    var val78 = data3_65536[(alu16+54)];
    var val79 = data3_65536[(alu16+55)];
    var val80 = data3_65536[(alu16+56)];
    var val81 = data3_65536[(alu16+57)];
    var val82 = data3_65536[(alu16+58)];
    var val83 = data3_65536[(alu16+59)];
    var val84 = data3_65536[(alu16+6)];
    var val85 = data3_65536[(alu16+60)];
    var val86 = data3_65536[(alu16+61)];
    var val87 = data3_65536[(alu16+62)];
    var val88 = data3_65536[(alu16+63)];
    var val89 = data3_65536[(alu16+64)];
    var val90 = data3_65536[(alu16+65)];
    var val91 = data3_65536[(alu16+66)];
    var val92 = data3_65536[(alu16+67)];
    var val93 = data3_65536[(alu16+68)];
    var val94 = data3_65536[(alu16+69)];
    var val95 = data3_65536[(alu16+7)];
    var val96 = data3_65536[(alu16+70)];
    var val97 = data3_65536[(alu16+71)];
    var val98 = data3_65536[(alu16+72)];
    var val99 = data3_65536[(alu16+73)];
    var val100 = data3_65536[(alu16+74)];
    var val101 = data3_65536[(alu16+75)];
    var val102 = data3_65536[(alu16+76)];
    var val103 = data3_65536[(alu16+77)];
    var val104 = data3_65536[(alu16+78)];
    var val105 = data3_65536[(alu16+79)];
    var val106 = data3_65536[(alu16+8)];
    var val107 = data3_65536[(alu16+80)];
    var val108 = data3_65536[(alu16+81)];
    var val109 = data3_65536[(alu16+82)];
    var val110 = data3_65536[(alu16+83)];
    var val111 = data3_65536[(alu16+84)];
    var val112 = data3_65536[(alu16+85)];
    var val113 = data3_65536[(alu16+86)];
    var val114 = data3_65536[(alu16+87)];
    var val115 = data3_65536[(alu16+88)];
    var val116 = data3_65536[(alu16+89)];
    var val117 = data3_65536[(alu16+9)];
    var val118 = data3_65536[(alu16+90)];
    var val119 = data3_65536[(alu16+91)];
    var val120 = data3_65536[(alu16+92)];
    var val121 = data3_65536[(alu16+93)];
    var val122 = data3_65536[(alu16+94)];
    var val123 = data3_65536[(alu16+95)];
    var val124 = data3_65536[(alu16+96)];
    var val125 = data3_65536[(alu16+97)];
    var val126 = data3_65536[(alu16+98)];
    var val127 = data3_65536[(alu16+99)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast4<<5u)));
    var val128 = data4_65536[alu17];
    var val129 = data4_65536[(alu17+1)];
    var val130 = data4_65536[(alu17+10)];
    var val131 = data4_65536[(alu17+11)];
    var val132 = data4_65536[(alu17+12)];
    var val133 = data4_65536[(alu17+13)];
    var val134 = data4_65536[(alu17+14)];
    var val135 = data4_65536[(alu17+15)];
    var val136 = data4_65536[(alu17+16)];
    var val137 = data4_65536[(alu17+17)];
    var val138 = data4_65536[(alu17+18)];
    var val139 = data4_65536[(alu17+19)];
    var val140 = data4_65536[(alu17+2)];
    var val141 = data4_65536[(alu17+20)];
    var val142 = data4_65536[(alu17+21)];
    var val143 = data4_65536[(alu17+22)];
    var val144 = data4_65536[(alu17+23)];
    var val145 = data4_65536[(alu17+24)];
    var val146 = data4_65536[(alu17+25)];
    var val147 = data4_65536[(alu17+256)];
    var val148 = data4_65536[(alu17+257)];
    var val149 = data4_65536[(alu17+258)];
    var val150 = data4_65536[(alu17+259)];
    var val151 = data4_65536[(alu17+26)];
    var val152 = data4_65536[(alu17+260)];
    var val153 = data4_65536[(alu17+261)];
    var val154 = data4_65536[(alu17+262)];
    var val155 = data4_65536[(alu17+263)];
    var val156 = data4_65536[(alu17+264)];
    var val157 = data4_65536[(alu17+265)];
    var val158 = data4_65536[(alu17+266)];
    var val159 = data4_65536[(alu17+267)];
    var val160 = data4_65536[(alu17+268)];
    var val161 = data4_65536[(alu17+269)];
    var val162 = data4_65536[(alu17+27)];
    var val163 = data4_65536[(alu17+270)];
    var val164 = data4_65536[(alu17+271)];
    var val165 = data4_65536[(alu17+272)];
    var val166 = data4_65536[(alu17+273)];
    var val167 = data4_65536[(alu17+274)];
    var val168 = data4_65536[(alu17+275)];
    var val169 = data4_65536[(alu17+276)];
    var val170 = data4_65536[(alu17+277)];
    var val171 = data4_65536[(alu17+278)];
    var val172 = data4_65536[(alu17+279)];
    var val173 = data4_65536[(alu17+28)];
    var val174 = data4_65536[(alu17+280)];
    var val175 = data4_65536[(alu17+281)];
    var val176 = data4_65536[(alu17+282)];
    var val177 = data4_65536[(alu17+283)];
    var val178 = data4_65536[(alu17+284)];
    var val179 = data4_65536[(alu17+285)];
    var val180 = data4_65536[(alu17+286)];
    var val181 = data4_65536[(alu17+287)];
    var val182 = data4_65536[(alu17+29)];
    var val183 = data4_65536[(alu17+3)];
    var val184 = data4_65536[(alu17+30)];
    var val185 = data4_65536[(alu17+31)];
    var val186 = data4_65536[(alu17+4)];
    var val187 = data4_65536[(alu17+5)];
    var val188 = data4_65536[(alu17+512)];
    var val189 = data4_65536[(alu17+513)];
    var val190 = data4_65536[(alu17+514)];
    var val191 = data4_65536[(alu17+515)];
    var val192 = data4_65536[(alu17+516)];
    var val193 = data4_65536[(alu17+517)];
    var val194 = data4_65536[(alu17+518)];
    var val195 = data4_65536[(alu17+519)];
    var val196 = data4_65536[(alu17+520)];
    var val197 = data4_65536[(alu17+521)];
    var val198 = data4_65536[(alu17+522)];
    var val199 = data4_65536[(alu17+523)];
    var val200 = data4_65536[(alu17+524)];
    var val201 = data4_65536[(alu17+525)];
    var val202 = data4_65536[(alu17+526)];
    var val203 = data4_65536[(alu17+527)];
    var val204 = data4_65536[(alu17+528)];
    var val205 = data4_65536[(alu17+529)];
    var val206 = data4_65536[(alu17+530)];
    var val207 = data4_65536[(alu17+531)];
    var val208 = data4_65536[(alu17+532)];
    var val209 = data4_65536[(alu17+533)];
    var val210 = data4_65536[(alu17+534)];
    var val211 = data4_65536[(alu17+535)];
    var val212 = data4_65536[(alu17+536)];
    var val213 = data4_65536[(alu17+537)];
    var val214 = data4_65536[(alu17+538)];
    var val215 = data4_65536[(alu17+539)];
    var val216 = data4_65536[(alu17+540)];
    var val217 = data4_65536[(alu17+541)];
    var val218 = data4_65536[(alu17+542)];
    var val219 = data4_65536[(alu17+543)];
    var val220 = data4_65536[(alu17+6)];
    var val221 = data4_65536[(alu17+7)];
    var val222 = data4_65536[(alu17+768)];
    var val223 = data4_65536[(alu17+769)];
    var val224 = data4_65536[(alu17+770)];
    var val225 = data4_65536[(alu17+771)];
    var val226 = data4_65536[(alu17+772)];
    var val227 = data4_65536[(alu17+773)];
    var val228 = data4_65536[(alu17+774)];
    var val229 = data4_65536[(alu17+775)];
    var val230 = data4_65536[(alu17+776)];
    var val231 = data4_65536[(alu17+777)];
    var val232 = data4_65536[(alu17+778)];
    var val233 = data4_65536[(alu17+779)];
    var val234 = data4_65536[(alu17+780)];
    var val235 = data4_65536[(alu17+781)];
    var val236 = data4_65536[(alu17+782)];
    var val237 = data4_65536[(alu17+783)];
    var val238 = data4_65536[(alu17+784)];
    var val239 = data4_65536[(alu17+785)];
    var val240 = data4_65536[(alu17+786)];
    var val241 = data4_65536[(alu17+787)];
    var val242 = data4_65536[(alu17+788)];
    var val243 = data4_65536[(alu17+789)];
    var val244 = data4_65536[(alu17+790)];
    var val245 = data4_65536[(alu17+791)];
    var val246 = data4_65536[(alu17+792)];
    var val247 = data4_65536[(alu17+793)];
    var val248 = data4_65536[(alu17+794)];
    var val249 = data4_65536[(alu17+795)];
    var val250 = data4_65536[(alu17+796)];
    var val251 = data4_65536[(alu17+797)];
    var val252 = data4_65536[(alu17+798)];
    var val253 = data4_65536[(alu17+799)];
    var val254 = data4_65536[(alu17+8)];
    var val255 = data4_65536[(alu17+9)];
    buf0[0] = (buf0[0]+(val0*val128)+(val1*val129)+(val40*val140)+(val51*val183)+(val62*val186)+(val73*val187)+(val84*val220)+(val95*val221)+(val106*val254)+(val117*val255)+(val2*val130)+(val13*val131)+(val24*val132)+(val33*val133)+(val34*val134)+(val35*val135)+(val36*val136)+(val37*val137)+(val38*val138)+(val39*val139)+(val41*val141)+(val42*val142)+(val43*val143)+(val44*val144)+(val45*val145)+(val46*val146)+(val47*val151)+(val48*val162)+(val49*val173)+(val50*val182)+(val52*val184)+(val53*val185));
    buf0[1] = (buf0[1]+(val0*val147)+(val1*val148)+(val40*val149)+(val51*val150)+(val62*val152)+(val73*val153)+(val84*val154)+(val95*val155)+(val106*val156)+(val117*val157)+(val2*val158)+(val13*val159)+(val24*val160)+(val33*val161)+(val34*val163)+(val35*val164)+(val36*val165)+(val37*val166)+(val38*val167)+(val39*val168)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val174)+(val46*val175)+(val47*val176)+(val48*val177)+(val49*val178)+(val50*val179)+(val52*val180)+(val53*val181));
    buf0[10] = (buf0[10]+(val89*val188)+(val90*val189)+(val91*val190)+(val92*val191)+(val93*val192)+(val94*val193)+(val96*val194)+(val97*val195)+(val98*val196)+(val99*val197)+(val100*val198)+(val101*val199)+(val102*val200)+(val103*val201)+(val104*val202)+(val105*val203)+(val107*val204)+(val108*val205)+(val109*val206)+(val110*val207)+(val111*val208)+(val112*val209)+(val113*val210)+(val114*val211)+(val115*val212)+(val116*val213)+(val118*val214)+(val119*val215)+(val120*val216)+(val121*val217)+(val122*val218)+(val123*val219));
    buf0[11] = (buf0[11]+(val89*val222)+(val90*val223)+(val91*val224)+(val92*val225)+(val93*val226)+(val94*val227)+(val96*val228)+(val97*val229)+(val98*val230)+(val99*val231)+(val100*val232)+(val101*val233)+(val102*val234)+(val103*val235)+(val104*val236)+(val105*val237)+(val107*val238)+(val108*val239)+(val109*val240)+(val110*val241)+(val111*val242)+(val112*val243)+(val113*val244)+(val114*val245)+(val115*val246)+(val116*val247)+(val118*val248)+(val119*val249)+(val120*val250)+(val121*val251)+(val122*val252)+(val123*val253));
    buf0[12] = (buf0[12]+(val124*val128)+(val125*val129)+(val126*val140)+(val127*val183)+(val3*val186)+(val4*val187)+(val5*val220)+(val6*val221)+(val7*val254)+(val8*val255)+(val9*val130)+(val10*val131)+(val11*val132)+(val12*val133)+(val14*val134)+(val15*val135)+(val16*val136)+(val17*val137)+(val18*val138)+(val19*val139)+(val20*val141)+(val21*val142)+(val22*val143)+(val23*val144)+(val25*val145)+(val26*val146)+(val27*val151)+(val28*val162)+(val29*val173)+(val30*val182)+(val31*val184)+(val32*val185));
    buf0[13] = (buf0[13]+(val124*val147)+(val125*val148)+(val126*val149)+(val127*val150)+(val3*val152)+(val4*val153)+(val5*val154)+(val6*val155)+(val7*val156)+(val8*val157)+(val9*val158)+(val10*val159)+(val11*val160)+(val12*val161)+(val14*val163)+(val15*val164)+(val16*val165)+(val17*val166)+(val18*val167)+(val19*val168)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val174)+(val26*val175)+(val27*val176)+(val28*val177)+(val29*val178)+(val30*val179)+(val31*val180)+(val32*val181));
    buf0[14] = (buf0[14]+(val124*val188)+(val125*val189)+(val126*val190)+(val127*val191)+(val3*val192)+(val4*val193)+(val5*val194)+(val6*val195)+(val7*val196)+(val8*val197)+(val9*val198)+(val10*val199)+(val11*val200)+(val12*val201)+(val14*val202)+(val15*val203)+(val16*val204)+(val17*val205)+(val18*val206)+(val19*val207)+(val20*val208)+(val21*val209)+(val22*val210)+(val23*val211)+(val25*val212)+(val26*val213)+(val27*val214)+(val28*val215)+(val29*val216)+(val30*val217)+(val31*val218)+(val32*val219));
    buf0[15] = (buf0[15]+(val124*val222)+(val125*val223)+(val126*val224)+(val127*val225)+(val3*val226)+(val4*val227)+(val5*val228)+(val6*val229)+(val7*val230)+(val8*val231)+(val9*val232)+(val10*val233)+(val11*val234)+(val12*val235)+(val14*val236)+(val15*val237)+(val16*val238)+(val17*val239)+(val18*val240)+(val19*val241)+(val20*val242)+(val21*val243)+(val22*val244)+(val23*val245)+(val25*val246)+(val26*val247)+(val27*val248)+(val28*val249)+(val29*val250)+(val30*val251)+(val31*val252)+(val32*val253));
    buf0[2] = (buf0[2]+(val0*val188)+(val1*val189)+(val40*val190)+(val51*val191)+(val62*val192)+(val73*val193)+(val84*val194)+(val95*val195)+(val106*val196)+(val117*val197)+(val2*val198)+(val13*val199)+(val24*val200)+(val33*val201)+(val34*val202)+(val35*val203)+(val36*val204)+(val37*val205)+(val38*val206)+(val39*val207)+(val41*val208)+(val42*val209)+(val43*val210)+(val44*val211)+(val45*val212)+(val46*val213)+(val47*val214)+(val48*val215)+(val49*val216)+(val50*val217)+(val52*val218)+(val53*val219));
    buf0[3] = (buf0[3]+(val0*val222)+(val1*val223)+(val40*val224)+(val51*val225)+(val62*val226)+(val73*val227)+(val84*val228)+(val95*val229)+(val106*val230)+(val117*val231)+(val2*val232)+(val13*val233)+(val24*val234)+(val33*val235)+(val34*val236)+(val35*val237)+(val36*val238)+(val37*val239)+(val38*val240)+(val39*val241)+(val41*val242)+(val42*val243)+(val43*val244)+(val44*val245)+(val45*val246)+(val46*val247)+(val47*val248)+(val48*val249)+(val49*val250)+(val50*val251)+(val52*val252)+(val53*val253));
    buf0[4] = (buf0[4]+(val54*val128)+(val55*val129)+(val56*val140)+(val57*val183)+(val58*val186)+(val59*val187)+(val60*val220)+(val61*val221)+(val63*val254)+(val64*val255)+(val65*val130)+(val66*val131)+(val67*val132)+(val68*val133)+(val69*val134)+(val70*val135)+(val71*val136)+(val72*val137)+(val74*val138)+(val75*val139)+(val76*val141)+(val77*val142)+(val78*val143)+(val79*val144)+(val80*val145)+(val81*val146)+(val82*val151)+(val83*val162)+(val85*val173)+(val86*val182)+(val87*val184)+(val88*val185));
    buf0[5] = (buf0[5]+(val54*val147)+(val55*val148)+(val56*val149)+(val57*val150)+(val58*val152)+(val59*val153)+(val60*val154)+(val61*val155)+(val63*val156)+(val64*val157)+(val65*val158)+(val66*val159)+(val67*val160)+(val68*val161)+(val69*val163)+(val70*val164)+(val71*val165)+(val72*val166)+(val74*val167)+(val75*val168)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val174)+(val81*val175)+(val82*val176)+(val83*val177)+(val85*val178)+(val86*val179)+(val87*val180)+(val88*val181));
    buf0[6] = (buf0[6]+(val54*val188)+(val55*val189)+(val56*val190)+(val57*val191)+(val58*val192)+(val59*val193)+(val60*val194)+(val61*val195)+(val63*val196)+(val64*val197)+(val65*val198)+(val66*val199)+(val67*val200)+(val68*val201)+(val69*val202)+(val70*val203)+(val71*val204)+(val72*val205)+(val74*val206)+(val75*val207)+(val76*val208)+(val77*val209)+(val78*val210)+(val79*val211)+(val80*val212)+(val81*val213)+(val82*val214)+(val83*val215)+(val85*val216)+(val86*val217)+(val87*val218)+(val88*val219));
    buf0[7] = (buf0[7]+(val54*val222)+(val55*val223)+(val56*val224)+(val57*val225)+(val58*val226)+(val59*val227)+(val60*val228)+(val61*val229)+(val63*val230)+(val64*val231)+(val65*val232)+(val66*val233)+(val67*val234)+(val68*val235)+(val69*val236)+(val70*val237)+(val71*val238)+(val72*val239)+(val74*val240)+(val75*val241)+(val76*val242)+(val77*val243)+(val78*val244)+(val79*val245)+(val80*val246)+(val81*val247)+(val82*val248)+(val83*val249)+(val85*val250)+(val86*val251)+(val87*val252)+(val88*val253));
    buf0[8] = (buf0[8]+(val89*val128)+(val90*val129)+(val91*val140)+(val92*val183)+(val93*val186)+(val94*val187)+(val96*val220)+(val97*val221)+(val98*val254)+(val99*val255)+(val100*val130)+(val101*val131)+(val102*val132)+(val103*val133)+(val104*val134)+(val105*val135)+(val107*val136)+(val108*val137)+(val109*val138)+(val110*val139)+(val111*val141)+(val112*val142)+(val113*val143)+(val114*val144)+(val115*val145)+(val116*val146)+(val118*val151)+(val119*val162)+(val120*val173)+(val121*val182)+(val122*val184)+(val123*val185));
    buf0[9] = (buf0[9]+(val89*val147)+(val90*val148)+(val91*val149)+(val92*val150)+(val93*val152)+(val94*val153)+(val96*val154)+(val97*val155)+(val98*val156)+(val99*val157)+(val100*val158)+(val101*val159)+(val102*val160)+(val103*val161)+(val104*val163)+(val105*val164)+(val107*val165)+(val108*val166)+(val109*val167)+(val110*val168)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val174)+(val116*val175)+(val118*val176)+(val119*val177)+(val120*val178)+(val121*val179)+(val122*val180)+(val123*val181));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<13u))+bitcast<i32>((cast2<<10u)));
  var val256 = data1_65536[alu36];
  var alu37 = (alu36+1);
  var val257 = data1_65536[alu37];
  var alu38 = (alu36+2);
  var val258 = data1_65536[alu38];
  var alu39 = (alu36+256);
  var val259 = data1_65536[alu39];
  var alu40 = (alu36+257);
  var val260 = data1_65536[alu40];
  var alu41 = (alu36+258);
  var val261 = data1_65536[alu41];
  var alu42 = (alu36+259);
  var val262 = data1_65536[alu42];
  var alu43 = (alu36+3);
  var val263 = data1_65536[alu43];
  var alu44 = (alu36+512);
  var val264 = data1_65536[alu44];
  var alu45 = (alu36+513);
  var val265 = data1_65536[alu45];
  var alu46 = (alu36+514);
  var val266 = data1_65536[alu46];
  var alu47 = (alu36+515);
  var val267 = data1_65536[alu47];
  var alu48 = (alu36+768);
  var val268 = data1_65536[alu48];
  var alu49 = (alu36+769);
  var val269 = data1_65536[alu49];
  var alu50 = (alu36+770);
  var val270 = data1_65536[alu50];
  var alu51 = (alu36+771);
  var val271 = data1_65536[alu51];
  var val272 = data2_1536[(alu35+512)];
  var val273 = data2_1536[(alu35+513)];
  var val274 = data2_1536[(alu35+514)];
  var val275 = data2_1536[(alu35+515)];
  var val276 = data5_256[alu35];
  var val277 = data5_256[(alu35+1)];
  var val278 = data5_256[(alu35+2)];
  var val279 = data5_256[(alu35+3)];
  data0_65536[alu36] = (val256+(val272*(buf0[0]+val276)));
  data0_65536[alu37] = (val257+(val273*(buf0[1]+val277)));
  data0_65536[alu38] = (val258+(val274*(buf0[2]+val278)));
  data0_65536[alu39] = (val259+(val272*(buf0[4]+val276)));
  data0_65536[alu40] = (val260+(val273*(buf0[5]+val277)));
  data0_65536[alu41] = (val261+(val274*(buf0[6]+val278)));
  data0_65536[alu42] = (val262+(val275*(buf0[7]+val279)));
  data0_65536[alu43] = (val263+(val275*(buf0[3]+val279)));
  data0_65536[alu44] = (val264+(val272*(buf0[8]+val276)));
  data0_65536[alu45] = (val265+(val273*(buf0[9]+val277)));
  data0_65536[alu46] = (val266+(val274*(buf0[10]+val278)));
  data0_65536[alu47] = (val267+(val275*(buf0[11]+val279)));
  data0_65536[alu48] = (val268+(val272*(buf0[12]+val276)));
  data0_65536[alu49] = (val269+(val273*(buf0[13]+val277)));
  data0_65536[alu50] = (val270+(val274*(buf0[14]+val278)));
  data0_65536[alu51] = (val271+(val275*(buf0[15]+val279)));
}`;

const r_16_8_4_4_32_dispatch_121 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_122 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var val21 = data3_256[(alu0+1)];
  var val22 = data3_256[(alu0+2)];
  var val23 = data3_256[(alu0+3)];
  var val24 = data4_1536[(alu0+1024)];
  var val25 = data4_1536[(alu0+1025)];
  var val26 = data4_1536[(alu0+1026)];
  var val27 = data4_1536[(alu0+1027)];
  var val28 = data4_1536[(alu0+768)];
  var val29 = data4_1536[(alu0+769)];
  var val30 = data4_1536[(alu0+770)];
  var val31 = data4_1536[(alu0+771)];
  var alu18 = (1/val16);
  var alu19 = (1/val17);
  var alu20 = (1/val18);
  var alu21 = (1/val19);
  var alu22 = (1.0f+val24);
  var alu23 = (1.0f+val25);
  var alu24 = (1.0f+val26);
  var alu25 = (1.0f+val27);
  data0_65536[alu1] = ((val0*alu18*val20*alu22)+val28);
  data0_65536[alu2] = ((val1*alu18*val21*alu23)+val29);
  data0_65536[alu3] = ((val2*alu18*val22*alu24)+val30);
  data0_65536[alu4] = ((val3*alu19*val20*alu22)+val28);
  data0_65536[alu5] = ((val4*alu19*val21*alu23)+val29);
  data0_65536[alu6] = ((val5*alu19*val22*alu24)+val30);
  data0_65536[alu7] = ((val6*alu19*val23*alu25)+val31);
  data0_65536[alu8] = ((val7*alu18*val23*alu25)+val31);
  data0_65536[alu9] = ((val8*alu20*val20*alu22)+val28);
  data0_65536[alu10] = ((val9*alu20*val21*alu23)+val29);
  data0_65536[alu11] = ((val10*alu20*val22*alu24)+val30);
  data0_65536[alu12] = ((val11*alu20*val23*alu25)+val31);
  data0_65536[alu13] = ((val12*alu21*val20*alu22)+val28);
  data0_65536[alu14] = ((val13*alu21*val21*alu23)+val29);
  data0_65536[alu15] = ((val14*alu21*val22*alu24)+val30);
  data0_65536[alu16] = ((val15*alu21*val23*alu25)+val31);
}`;

const r_8_22_8_16_4_4_64_4_dispatch_123 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_360448:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_360448:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1408:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 22 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast2 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast2);
    var val0 = data1_65536[alu16];
    var val1 = data1_65536[(alu16+1)];
    var val2 = data1_65536[(alu16+2)];
    var val3 = data1_65536[(alu16+256)];
    var val4 = data1_65536[(alu16+257)];
    var val5 = data1_65536[(alu16+258)];
    var val6 = data1_65536[(alu16+259)];
    var val7 = data1_65536[(alu16+3)];
    var val8 = data1_65536[(alu16+512)];
    var val9 = data1_65536[(alu16+513)];
    var val10 = data1_65536[(alu16+514)];
    var val11 = data1_65536[(alu16+515)];
    var val12 = data1_65536[(alu16+768)];
    var val13 = data1_65536[(alu16+769)];
    var val14 = data1_65536[(alu16+770)];
    var val15 = data1_65536[(alu16+771)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast1<<10u))+cast2);
    var val16 = data2_360448[alu17];
    var val17 = data2_360448[(alu17+1)];
    var val18 = data2_360448[(alu17+2)];
    var val19 = data2_360448[(alu17+256)];
    var val20 = data2_360448[(alu17+257)];
    var val21 = data2_360448[(alu17+258)];
    var val22 = data2_360448[(alu17+259)];
    var val23 = data2_360448[(alu17+3)];
    var val24 = data2_360448[(alu17+512)];
    var val25 = data2_360448[(alu17+513)];
    var val26 = data2_360448[(alu17+514)];
    var val27 = data2_360448[(alu17+515)];
    var val28 = data2_360448[(alu17+768)];
    var val29 = data2_360448[(alu17+769)];
    var val30 = data2_360448[(alu17+770)];
    var val31 = data2_360448[(alu17+771)];
    buf0[0] = (buf0[0]+(val0*val16)+(val1*val17)+(val2*val18)+(val7*val23));
    buf0[1] = (buf0[1]+(val0*val19)+(val1*val20)+(val2*val21)+(val7*val22));
    buf0[10] = (buf0[10]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    buf0[11] = (buf0[11]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    buf0[12] = (buf0[12]+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val23));
    buf0[13] = (buf0[13]+(val12*val19)+(val13*val20)+(val14*val21)+(val15*val22));
    buf0[14] = (buf0[14]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    buf0[15] = (buf0[15]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    buf0[2] = (buf0[2]+(val0*val24)+(val1*val25)+(val2*val26)+(val7*val27));
    buf0[3] = (buf0[3]+(val0*val28)+(val1*val29)+(val2*val30)+(val7*val31));
    buf0[4] = (buf0[4]+(val3*val16)+(val4*val17)+(val5*val18)+(val6*val23));
    buf0[5] = (buf0[5]+(val3*val19)+(val4*val20)+(val5*val21)+(val6*val22));
    buf0[6] = (buf0[6]+(val3*val24)+(val4*val25)+(val5*val26)+(val6*val27));
    buf0[7] = (buf0[7]+(val3*val28)+(val4*val29)+(val5*val30)+(val6*val31));
    buf0[8] = (buf0[8]+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val23));
    buf0[9] = (buf0[9]+(val8*val19)+(val9*val20)+(val10*val21)+(val11*val22));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)));
  var val32 = data3_1408[alu35];
  var val33 = data3_1408[(alu35+1)];
  var val34 = data3_1408[(alu35+2)];
  var val35 = data3_1408[(alu35+3)];
  var alu36 = (alu35+(gidx1*45056)+(lidx0*5632));
  data0_360448[alu36] = (buf0[0]+val32);
  data0_360448[(alu36+1)] = (buf0[1]+val33);
  data0_360448[(alu36+1408)] = (buf0[4]+val32);
  data0_360448[(alu36+1409)] = (buf0[5]+val33);
  data0_360448[(alu36+1410)] = (buf0[6]+val34);
  data0_360448[(alu36+1411)] = (buf0[7]+val35);
  data0_360448[(alu36+2)] = (buf0[2]+val34);
  data0_360448[(alu36+2816)] = (buf0[8]+val32);
  data0_360448[(alu36+2817)] = (buf0[9]+val33);
  data0_360448[(alu36+2818)] = (buf0[10]+val34);
  data0_360448[(alu36+2819)] = (buf0[11]+val35);
  data0_360448[(alu36+3)] = (buf0[3]+val35);
  data0_360448[(alu36+4224)] = (buf0[12]+val32);
  data0_360448[(alu36+4225)] = (buf0[13]+val33);
  data0_360448[(alu36+4226)] = (buf0[14]+val34);
  data0_360448[(alu36+4227)] = (buf0[15]+val35);
}`;

const r_8_4_8_16_4_4_176_4_dispatch_124 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_360448:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_180224:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 176; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = ((gidx1*45056)+(lidx0*5632)+cast0);
    var val0 = data3_360448[(alu16+1)];
    var val1 = data3_360448[(alu16+1408)];
    var val2 = data3_360448[(alu16+1409)];
    var val3 = data3_360448[(alu16+1410)];
    var val4 = data3_360448[(alu16+1411)];
    var val5 = data3_360448[(alu16+2)];
    var val6 = data3_360448[(alu16+2112)];
    var val7 = data3_360448[(alu16+2113)];
    var val8 = data3_360448[(alu16+2114)];
    var val9 = data3_360448[(alu16+2115)];
    var val10 = data3_360448[(alu16+2816)];
    var val11 = data3_360448[(alu16+2817)];
    var val12 = data3_360448[(alu16+2818)];
    var val13 = data3_360448[(alu16+2819)];
    var val14 = data3_360448[(alu16+3)];
    var val15 = data3_360448[(alu16+3520)];
    var val16 = data3_360448[(alu16+3521)];
    var val17 = data3_360448[(alu16+3522)];
    var val18 = data3_360448[(alu16+3523)];
    var val19 = data3_360448[(alu16+4224)];
    var val20 = data3_360448[(alu16+4225)];
    var val21 = data3_360448[(alu16+4226)];
    var val22 = data3_360448[(alu16+4227)];
    var val23 = data3_360448[(alu16+4928)];
    var val24 = data3_360448[(alu16+4929)];
    var val25 = data3_360448[(alu16+4930)];
    var val26 = data3_360448[(alu16+4931)];
    var val27 = data3_360448[(alu16+704)];
    var val28 = data3_360448[(alu16+705)];
    var val29 = data3_360448[(alu16+706)];
    var val30 = data3_360448[(alu16+707)];
    var val31 = data3_360448[alu16];
    var alu17 = ((gidx0*45056)+(lidx1*2816)+cast0);
    var val32 = data4_180224[(alu17+1)];
    var val33 = data4_180224[(alu17+1408)];
    var val34 = data4_180224[(alu17+1409)];
    var val35 = data4_180224[(alu17+1410)];
    var val36 = data4_180224[(alu17+1411)];
    var val37 = data4_180224[(alu17+2)];
    var val38 = data4_180224[(alu17+2112)];
    var val39 = data4_180224[(alu17+2113)];
    var val40 = data4_180224[(alu17+2114)];
    var val41 = data4_180224[(alu17+2115)];
    var val42 = data4_180224[(alu17+3)];
    var val43 = data4_180224[(alu17+704)];
    var val44 = data4_180224[(alu17+705)];
    var val45 = data4_180224[(alu17+706)];
    var val46 = data4_180224[(alu17+707)];
    var val47 = data4_180224[alu17];
    var alu18 = (val0*(1/(1.0f+exp2((val0*-1.4426950216293335f))))*val28);
    var alu19 = (val31*(1/(1.0f+exp2((val31*-1.4426950216293335f))))*val27);
    var alu20 = (val5*(1/(1.0f+exp2((val5*-1.4426950216293335f))))*val29);
    var alu21 = (val14*(1/(1.0f+exp2((val14*-1.4426950216293335f))))*val30);
    var alu22 = (val10*(1/(1.0f+exp2((val10*-1.4426950216293335f))))*val15);
    var alu23 = (val11*(1/(1.0f+exp2((val11*-1.4426950216293335f))))*val16);
    var alu24 = (val12*(1/(1.0f+exp2((val12*-1.4426950216293335f))))*val17);
    var alu25 = (val13*(1/(1.0f+exp2((val13*-1.4426950216293335f))))*val18);
    var alu26 = (val19*(1/(1.0f+exp2((val19*-1.4426950216293335f))))*val23);
    var alu27 = (val20*(1/(1.0f+exp2((val20*-1.4426950216293335f))))*val24);
    var alu28 = (val21*(1/(1.0f+exp2((val21*-1.4426950216293335f))))*val25);
    var alu29 = (val22*(1/(1.0f+exp2((val22*-1.4426950216293335f))))*val26);
    var alu30 = (val1*(1/(1.0f+exp2((val1*-1.4426950216293335f))))*val6);
    var alu31 = (val2*(1/(1.0f+exp2((val2*-1.4426950216293335f))))*val7);
    var alu32 = (val3*(1/(1.0f+exp2((val3*-1.4426950216293335f))))*val8);
    var alu33 = (val4*(1/(1.0f+exp2((val4*-1.4426950216293335f))))*val9);
    buf0[0] = (buf0[0]+(alu19*val47)+(alu18*val32)+(alu20*val37)+(alu21*val42));
    buf0[1] = (buf0[1]+(alu19*val43)+(alu18*val44)+(alu20*val45)+(alu21*val46));
    buf0[10] = (buf0[10]+(alu22*val33)+(alu23*val34)+(alu24*val35)+(alu25*val36));
    buf0[11] = (buf0[11]+(alu22*val38)+(alu23*val39)+(alu24*val40)+(alu25*val41));
    buf0[12] = (buf0[12]+(alu26*val47)+(alu27*val32)+(alu28*val37)+(alu29*val42));
    buf0[13] = (buf0[13]+(alu26*val43)+(alu27*val44)+(alu28*val45)+(alu29*val46));
    buf0[14] = (buf0[14]+(alu26*val33)+(alu27*val34)+(alu28*val35)+(alu29*val36));
    buf0[15] = (buf0[15]+(alu26*val38)+(alu27*val39)+(alu28*val40)+(alu29*val41));
    buf0[2] = (buf0[2]+(alu19*val33)+(alu18*val34)+(alu20*val35)+(alu21*val36));
    buf0[3] = (buf0[3]+(alu19*val38)+(alu18*val39)+(alu20*val40)+(alu21*val41));
    buf0[4] = (buf0[4]+(alu30*val47)+(alu31*val32)+(alu32*val37)+(alu33*val42));
    buf0[5] = (buf0[5]+(alu30*val43)+(alu31*val44)+(alu32*val45)+(alu33*val46));
    buf0[6] = (buf0[6]+(alu30*val33)+(alu31*val34)+(alu32*val35)+(alu33*val36));
    buf0[7] = (buf0[7]+(alu30*val38)+(alu31*val39)+(alu32*val40)+(alu33*val41));
    buf0[8] = (buf0[8]+(alu22*val47)+(alu23*val32)+(alu24*val37)+(alu25*val42));
    buf0[9] = (buf0[9]+(alu22*val43)+(alu23*val44)+(alu24*val45)+(alu25*val46));
  }
  var alu51 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu52 = (alu51+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u)));
  var val48 = data1_65536[alu52];
  var alu53 = (alu52+1);
  var val49 = data1_65536[alu53];
  var alu54 = (alu52+2);
  var val50 = data1_65536[alu54];
  var alu55 = (alu52+256);
  var val51 = data1_65536[alu55];
  var alu56 = (alu52+257);
  var val52 = data1_65536[alu56];
  var alu57 = (alu52+258);
  var val53 = data1_65536[alu57];
  var alu58 = (alu52+259);
  var val54 = data1_65536[alu58];
  var alu59 = (alu52+3);
  var val55 = data1_65536[alu59];
  var alu60 = (alu52+512);
  var val56 = data1_65536[alu60];
  var alu61 = (alu52+513);
  var val57 = data1_65536[alu61];
  var alu62 = (alu52+514);
  var val58 = data1_65536[alu62];
  var alu63 = (alu52+515);
  var val59 = data1_65536[alu63];
  var alu64 = (alu52+768);
  var val60 = data1_65536[alu64];
  var alu65 = (alu52+769);
  var val61 = data1_65536[alu65];
  var alu66 = (alu52+770);
  var val62 = data1_65536[alu66];
  var alu67 = (alu52+771);
  var val63 = data1_65536[alu67];
  var val64 = data2_1536[(alu51+1280)];
  var val65 = data2_1536[(alu51+1281)];
  var val66 = data2_1536[(alu51+1282)];
  var val67 = data2_1536[(alu51+1283)];
  var val68 = data5_256[alu51];
  var val69 = data5_256[(alu51+1)];
  var val70 = data5_256[(alu51+2)];
  var val71 = data5_256[(alu51+3)];
  data0_65536[alu52] = (val48+(val64*(buf0[0]+val68)));
  data0_65536[alu53] = (val49+(val65*(buf0[1]+val69)));
  data0_65536[alu54] = (val50+(val66*(buf0[2]+val70)));
  data0_65536[alu55] = (val51+(val64*(buf0[4]+val68)));
  data0_65536[alu56] = (val52+(val65*(buf0[5]+val69)));
  data0_65536[alu57] = (val53+(val66*(buf0[6]+val70)));
  data0_65536[alu58] = (val54+(val67*(buf0[7]+val71)));
  data0_65536[alu59] = (val55+(val67*(buf0[3]+val71)));
  data0_65536[alu60] = (val56+(val64*(buf0[8]+val68)));
  data0_65536[alu61] = (val57+(val65*(buf0[9]+val69)));
  data0_65536[alu62] = (val58+(val66*(buf0[10]+val70)));
  data0_65536[alu63] = (val59+(val67*(buf0[11]+val71)));
  data0_65536[alu64] = (val60+(val64*(buf0[12]+val68)));
  data0_65536[alu65] = (val61+(val65*(buf0[13]+val69)));
  data0_65536[alu66] = (val62+(val66*(buf0[14]+val70)));
  data0_65536[alu67] = (val63+(val67*(buf0[15]+val71)));
}`;

const r_16_8_4_4_32_dispatch_125 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_126 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var alu18 = (alu0+1);
  var val21 = data3_256[alu18];
  var alu19 = (alu0+2);
  var val22 = data3_256[alu19];
  var alu20 = (alu0+3);
  var val23 = data3_256[alu20];
  var val24 = data4_1536[alu0];
  var val25 = data4_1536[alu18];
  var val26 = data4_1536[alu19];
  var val27 = data4_1536[(alu0+256)];
  var val28 = data4_1536[(alu0+257)];
  var val29 = data4_1536[(alu0+258)];
  var val30 = data4_1536[(alu0+259)];
  var val31 = data4_1536[alu20];
  var alu21 = (1/val16);
  var alu22 = (1/val17);
  var alu23 = (1/val18);
  var alu24 = (1/val19);
  var alu25 = (1.0f+val27);
  var alu26 = (1.0f+val28);
  var alu27 = (1.0f+val29);
  var alu28 = (1.0f+val30);
  data0_65536[alu1] = ((val0*alu21*val20*alu25)+val24);
  data0_65536[alu2] = ((val1*alu21*val21*alu26)+val25);
  data0_65536[alu3] = ((val2*alu21*val22*alu27)+val26);
  data0_65536[alu4] = ((val3*alu22*val20*alu25)+val24);
  data0_65536[alu5] = ((val4*alu22*val21*alu26)+val25);
  data0_65536[alu6] = ((val5*alu22*val22*alu27)+val26);
  data0_65536[alu7] = ((val6*alu22*val23*alu28)+val31);
  data0_65536[alu8] = ((val7*alu21*val23*alu28)+val31);
  data0_65536[alu9] = ((val8*alu23*val20*alu25)+val24);
  data0_65536[alu10] = ((val9*alu23*val21*alu26)+val25);
  data0_65536[alu11] = ((val10*alu23*val22*alu27)+val26);
  data0_65536[alu12] = ((val11*alu23*val23*alu28)+val31);
  data0_65536[alu13] = ((val12*alu24*val20*alu25)+val24);
  data0_65536[alu14] = ((val13*alu24*val21*alu26)+val25);
  data0_65536[alu15] = ((val14*alu24*val22*alu27)+val26);
  data0_65536[alu16] = ((val15*alu24*val23*alu28)+val31);
}`;

const r_8_16_8_16_3_4_64_4_dispatch_127 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_196608:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_196608:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_768:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,12>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu12 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast0);
    var val0 = data1_65536[alu12];
    var val1 = data1_65536[(alu12+1)];
    var val2 = data1_65536[(alu12+2)];
    var val3 = data1_65536[(alu12+256)];
    var val4 = data1_65536[(alu12+257)];
    var val5 = data1_65536[(alu12+258)];
    var val6 = data1_65536[(alu12+259)];
    var val7 = data1_65536[(alu12+3)];
    var val8 = data1_65536[(alu12+512)];
    var val9 = data1_65536[(alu12+513)];
    var val10 = data1_65536[(alu12+514)];
    var val11 = data1_65536[(alu12+515)];
    var val12 = data1_65536[(alu12+768)];
    var val13 = data1_65536[(alu12+769)];
    var val14 = data1_65536[(alu12+770)];
    var val15 = data1_65536[(alu12+771)];
    var alu13 = ((gidx0*12288)+(lidx1*768)+cast0);
    var val16 = data2_196608[(alu13+1)];
    var val17 = data2_196608[(alu13+2)];
    var val18 = data2_196608[(alu13+256)];
    var val19 = data2_196608[(alu13+257)];
    var val20 = data2_196608[(alu13+258)];
    var val21 = data2_196608[(alu13+259)];
    var val22 = data2_196608[(alu13+3)];
    var val23 = data2_196608[(alu13+512)];
    var val24 = data2_196608[(alu13+513)];
    var val25 = data2_196608[(alu13+514)];
    var val26 = data2_196608[(alu13+515)];
    var val27 = data2_196608[alu13];
    buf0[0] = (buf0[0]+(val0*val27)+(val1*val16)+(val2*val17)+(val7*val22));
    buf0[1] = (buf0[1]+(val0*val18)+(val1*val19)+(val2*val20)+(val7*val21));
    buf0[10] = (buf0[10]+(val12*val18)+(val13*val19)+(val14*val20)+(val15*val21));
    buf0[11] = (buf0[11]+(val12*val23)+(val13*val24)+(val14*val25)+(val15*val26));
    buf0[2] = (buf0[2]+(val0*val23)+(val1*val24)+(val2*val25)+(val7*val26));
    buf0[3] = (buf0[3]+(val3*val27)+(val4*val16)+(val5*val17)+(val6*val22));
    buf0[4] = (buf0[4]+(val3*val18)+(val4*val19)+(val5*val20)+(val6*val21));
    buf0[5] = (buf0[5]+(val3*val23)+(val4*val24)+(val5*val25)+(val6*val26));
    buf0[6] = (buf0[6]+(val8*val27)+(val9*val16)+(val10*val17)+(val11*val22));
    buf0[7] = (buf0[7]+(val8*val18)+(val9*val19)+(val10*val20)+(val11*val21));
    buf0[8] = (buf0[8]+(val8*val23)+(val9*val24)+(val10*val25)+(val11*val26));
    buf0[9] = (buf0[9]+(val12*val27)+(val13*val16)+(val14*val17)+(val15*val22));
  }
  var alu27 = ((gidx0*48)+(lidx1*3));
  var val28 = data3_768[(alu27+1)];
  var val29 = data3_768[(alu27+2)];
  var val30 = data3_768[alu27];
  var alu28 = (alu27+(gidx1*24576)+(lidx0*3072));
  data0_196608[(alu28+1)] = (buf0[1]+val28);
  data0_196608[(alu28+1536)] = (buf0[6]+val30);
  data0_196608[(alu28+1537)] = (buf0[7]+val28);
  data0_196608[(alu28+1538)] = (buf0[8]+val29);
  data0_196608[(alu28+2)] = (buf0[2]+val29);
  data0_196608[(alu28+2304)] = (buf0[9]+val30);
  data0_196608[(alu28+2305)] = (buf0[10]+val28);
  data0_196608[(alu28+2306)] = (buf0[11]+val29);
  data0_196608[(alu28+768)] = (buf0[3]+val30);
  data0_196608[(alu28+769)] = (buf0[4]+val28);
  data0_196608[(alu28+770)] = (buf0[5]+val29);
  data0_196608[alu28] = (buf0[0]+val30);
}`;

const r_8_16_8_4_4_4_dispatch_128 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[alu4];
    var val1 = data1_196608[(alu4+3072)];
    var val2 = data1_196608[(alu4+6144)];
    var val3 = data1_196608[(alu4+9216)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const r_8_16_8_4_4_4_dispatch_129 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 4; Lidx0++) {
    var alu4 = (bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0+bitcast<i32>((cast0<<5u))+(gidx0*12288)+(lidx1*768));
    var val0 = data1_196608[(alu4+256)];
    var val1 = data1_196608[(alu4+3328)];
    var val2 = data1_196608[(alu4+6400)];
    var val3 = data1_196608[(alu4+9472)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast1);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx103 = 0; Lidx103 < 8; Lidx103++) {
    var alu20 = (cast1+bitcast<i32>((bitcast<u32>(Lidx103)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((bitcast<u32>(gidx0)<<4u))+lidx1+bitcast<i32>((cast0<<8u)));
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_2048[alu26] = sqrt(((buf1[0]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+12)] = sqrt(((buf1[3]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+4)] = sqrt(((buf1[1]*0.03125f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_2048[(alu26+8)] = sqrt(((buf1[2]*0.03125f)+9.999999974752427e-07f));
  }
}`;

const E_2_32_8_16_2_4_dispatch_130 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+1)];
  var val1 = data1_196608[(alu0+32)];
  var val2 = data1_196608[(alu0+33)];
  var val3 = data1_196608[(alu0+64)];
  var val4 = data1_196608[(alu0+65)];
  var val5 = data1_196608[(alu0+96)];
  var val6 = data1_196608[(alu0+97)];
  var val7 = data1_196608[alu0];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val13);
  var alu10 = (val7*alu5*val12);
  var alu11 = (val1*alu6*val12);
  var alu12 = (val2*alu6*val13);
  var alu13 = (val3*alu7*val12);
  var alu14 = (val4*alu7*val13);
  var alu15 = (val5*alu8*val12);
  var alu16 = (val6*alu8*val13);
  data0_65536[alu4] = ((alu10*val14)-(alu9*val16));
  data0_65536[(alu4+1)] = ((alu9*val15)+(alu10*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const E_2_32_8_16_2_4_dispatch_131 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_196608:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_32:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_8192:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_8192:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 32 */
  var gidx1 = i32(gindex.y); /* 2 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<i32>((bitcast<u32>(lidx1)<<1u));
  var alu0 = ((gidx0*6144)+(lidx0*768)+bitcast<i32>((cast0<<7u))+cast1);
  var val0 = data1_196608[(alu0+256)];
  var val1 = data1_196608[(alu0+257)];
  var val2 = data1_196608[(alu0+288)];
  var val3 = data1_196608[(alu0+289)];
  var val4 = data1_196608[(alu0+320)];
  var val5 = data1_196608[(alu0+321)];
  var val6 = data1_196608[(alu0+352)];
  var val7 = data1_196608[(alu0+353)];
  var cast2 = bitcast<u32>(gidx0);
  var alu1 = (bitcast<i32>((cast2<<3u))+lidx0+bitcast<i32>((cast0<<10u)));
  var val8 = data2_2048[alu1];
  var val9 = data2_2048[(alu1+256)];
  var val10 = data2_2048[(alu1+512)];
  var val11 = data2_2048[(alu1+768)];
  var val12 = data3_32[cast1];
  var val13 = data3_32[(cast1+1)];
  var alu2 = (bitcast<i32>((cast2<<8u))+bitcast<i32>((bitcast<u32>(lidx0)<<5u))+cast1);
  var val14 = data4_8192[alu2];
  var alu3 = (alu2+1);
  var val15 = data4_8192[alu3];
  var val16 = data5_8192[alu2];
  var val17 = data5_8192[alu3];
  var alu4 = (alu2+bitcast<i32>((cast0<<15u)));
  var alu5 = (1/val8);
  var alu6 = (1/val9);
  var alu7 = (1/val10);
  var alu8 = (1/val11);
  var alu9 = (val0*alu5*val12);
  var alu10 = (val1*alu5*val13);
  var alu11 = (val2*alu6*val12);
  var alu12 = (val3*alu6*val13);
  var alu13 = (val4*alu7*val12);
  var alu14 = (val5*alu7*val13);
  var alu15 = (val6*alu8*val12);
  var alu16 = (val7*alu8*val13);
  data0_65536[alu4] = ((alu9*val14)-(alu10*val16));
  data0_65536[(alu4+1)] = ((alu10*val15)+(alu9*val17));
  data0_65536[(alu4+16384)] = ((alu13*val14)-(alu14*val16));
  data0_65536[(alu4+16385)] = ((alu14*val15)+(alu13*val17));
  data0_65536[(alu4+24576)] = ((alu15*val14)-(alu16*val16));
  data0_65536[(alu4+24577)] = ((alu16*val15)+(alu15*val17));
  data0_65536[(alu4+8192)] = ((alu11*val14)-(alu12*val16));
  data0_65536[(alu4+8193)] = ((alu12*val15)+(alu11*val17));
}`;

const r_8_8_4_8_16_4_4_32_dispatch_132 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_524288:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_65536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx1 = i32(gindex.y); /* 8 */
  var gidx2 = i32(gindex.z); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(gidx2);
  var cast2 = bitcast<i32>((cast1<<13u));
  var cast3 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<10u))+bitcast<i32>((cast3<<7u))+cast2);
  var val0 = data1_65536[alu0];
  var val1 = data1_65536[(alu0+1)];
  var val2 = data1_65536[(alu0+10)];
  var val3 = data1_65536[(alu0+100)];
  var val4 = data1_65536[(alu0+101)];
  var val5 = data1_65536[(alu0+102)];
  var val6 = data1_65536[(alu0+103)];
  var val7 = data1_65536[(alu0+104)];
  var val8 = data1_65536[(alu0+105)];
  var val9 = data1_65536[(alu0+106)];
  var val10 = data1_65536[(alu0+107)];
  var val11 = data1_65536[(alu0+108)];
  var val12 = data1_65536[(alu0+109)];
  var val13 = data1_65536[(alu0+11)];
  var val14 = data1_65536[(alu0+110)];
  var val15 = data1_65536[(alu0+111)];
  var val16 = data1_65536[(alu0+112)];
  var val17 = data1_65536[(alu0+113)];
  var val18 = data1_65536[(alu0+114)];
  var val19 = data1_65536[(alu0+115)];
  var val20 = data1_65536[(alu0+116)];
  var val21 = data1_65536[(alu0+117)];
  var val22 = data1_65536[(alu0+118)];
  var val23 = data1_65536[(alu0+119)];
  var val24 = data1_65536[(alu0+12)];
  var val25 = data1_65536[(alu0+120)];
  var val26 = data1_65536[(alu0+121)];
  var val27 = data1_65536[(alu0+122)];
  var val28 = data1_65536[(alu0+123)];
  var val29 = data1_65536[(alu0+124)];
  var val30 = data1_65536[(alu0+125)];
  var val31 = data1_65536[(alu0+126)];
  var val32 = data1_65536[(alu0+127)];
  var val33 = data1_65536[(alu0+13)];
  var val34 = data1_65536[(alu0+14)];
  var val35 = data1_65536[(alu0+15)];
  var val36 = data1_65536[(alu0+16)];
  var val37 = data1_65536[(alu0+17)];
  var val38 = data1_65536[(alu0+18)];
  var val39 = data1_65536[(alu0+19)];
  var val40 = data1_65536[(alu0+2)];
  var val41 = data1_65536[(alu0+20)];
  var val42 = data1_65536[(alu0+21)];
  var val43 = data1_65536[(alu0+22)];
  var val44 = data1_65536[(alu0+23)];
  var val45 = data1_65536[(alu0+24)];
  var val46 = data1_65536[(alu0+25)];
  var val47 = data1_65536[(alu0+26)];
  var val48 = data1_65536[(alu0+27)];
  var val49 = data1_65536[(alu0+28)];
  var val50 = data1_65536[(alu0+29)];
  var val51 = data1_65536[(alu0+3)];
  var val52 = data1_65536[(alu0+30)];
  var val53 = data1_65536[(alu0+31)];
  var val54 = data1_65536[(alu0+32)];
  var val55 = data1_65536[(alu0+33)];
  var val56 = data1_65536[(alu0+34)];
  var val57 = data1_65536[(alu0+35)];
  var val58 = data1_65536[(alu0+36)];
  var val59 = data1_65536[(alu0+37)];
  var val60 = data1_65536[(alu0+38)];
  var val61 = data1_65536[(alu0+39)];
  var val62 = data1_65536[(alu0+4)];
  var val63 = data1_65536[(alu0+40)];
  var val64 = data1_65536[(alu0+41)];
  var val65 = data1_65536[(alu0+42)];
  var val66 = data1_65536[(alu0+43)];
  var val67 = data1_65536[(alu0+44)];
  var val68 = data1_65536[(alu0+45)];
  var val69 = data1_65536[(alu0+46)];
  var val70 = data1_65536[(alu0+47)];
  var val71 = data1_65536[(alu0+48)];
  var val72 = data1_65536[(alu0+49)];
  var val73 = data1_65536[(alu0+5)];
  var val74 = data1_65536[(alu0+50)];
  var val75 = data1_65536[(alu0+51)];
  var val76 = data1_65536[(alu0+52)];
  var val77 = data1_65536[(alu0+53)];
  var val78 = data1_65536[(alu0+54)];
  var val79 = data1_65536[(alu0+55)];
  var val80 = data1_65536[(alu0+56)];
  var val81 = data1_65536[(alu0+57)];
  var val82 = data1_65536[(alu0+58)];
  var val83 = data1_65536[(alu0+59)];
  var val84 = data1_65536[(alu0+6)];
  var val85 = data1_65536[(alu0+60)];
  var val86 = data1_65536[(alu0+61)];
  var val87 = data1_65536[(alu0+62)];
  var val88 = data1_65536[(alu0+63)];
  var val89 = data1_65536[(alu0+64)];
  var val90 = data1_65536[(alu0+65)];
  var val91 = data1_65536[(alu0+66)];
  var val92 = data1_65536[(alu0+67)];
  var val93 = data1_65536[(alu0+68)];
  var val94 = data1_65536[(alu0+69)];
  var val95 = data1_65536[(alu0+7)];
  var val96 = data1_65536[(alu0+70)];
  var val97 = data1_65536[(alu0+71)];
  var val98 = data1_65536[(alu0+72)];
  var val99 = data1_65536[(alu0+73)];
  var val100 = data1_65536[(alu0+74)];
  var val101 = data1_65536[(alu0+75)];
  var val102 = data1_65536[(alu0+76)];
  var val103 = data1_65536[(alu0+77)];
  var val104 = data1_65536[(alu0+78)];
  var val105 = data1_65536[(alu0+79)];
  var val106 = data1_65536[(alu0+8)];
  var val107 = data1_65536[(alu0+80)];
  var val108 = data1_65536[(alu0+81)];
  var val109 = data1_65536[(alu0+82)];
  var val110 = data1_65536[(alu0+83)];
  var val111 = data1_65536[(alu0+84)];
  var val112 = data1_65536[(alu0+85)];
  var val113 = data1_65536[(alu0+86)];
  var val114 = data1_65536[(alu0+87)];
  var val115 = data1_65536[(alu0+88)];
  var val116 = data1_65536[(alu0+89)];
  var val117 = data1_65536[(alu0+9)];
  var val118 = data1_65536[(alu0+90)];
  var val119 = data1_65536[(alu0+91)];
  var val120 = data1_65536[(alu0+92)];
  var val121 = data1_65536[(alu0+93)];
  var val122 = data1_65536[(alu0+94)];
  var val123 = data1_65536[(alu0+95)];
  var val124 = data1_65536[(alu0+96)];
  var val125 = data1_65536[(alu0+97)];
  var val126 = data1_65536[(alu0+98)];
  var val127 = data1_65536[(alu0+99)];
  var gidx0 = i32(gindex.x); /* 4 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast4 = bitcast<u32>(gidx0);
  var cast5 = bitcast<u32>(lidx1);
  var alu1 = (bitcast<i32>((cast4<<11u))+bitcast<i32>((cast5<<7u))+cast2);
  var val128 = data2_65536[alu1];
  var val129 = data2_65536[(alu1+1)];
  var val130 = data2_65536[(alu1+10)];
  var val131 = data2_65536[(alu1+100)];
  var val132 = data2_65536[(alu1+101)];
  var val133 = data2_65536[(alu1+102)];
  var val134 = data2_65536[(alu1+103)];
  var val135 = data2_65536[(alu1+104)];
  var val136 = data2_65536[(alu1+105)];
  var val137 = data2_65536[(alu1+106)];
  var val138 = data2_65536[(alu1+107)];
  var val139 = data2_65536[(alu1+108)];
  var val140 = data2_65536[(alu1+109)];
  var val141 = data2_65536[(alu1+11)];
  var val142 = data2_65536[(alu1+110)];
  var val143 = data2_65536[(alu1+111)];
  var val144 = data2_65536[(alu1+112)];
  var val145 = data2_65536[(alu1+113)];
  var val146 = data2_65536[(alu1+114)];
  var val147 = data2_65536[(alu1+115)];
  var val148 = data2_65536[(alu1+116)];
  var val149 = data2_65536[(alu1+117)];
  var val150 = data2_65536[(alu1+118)];
  var val151 = data2_65536[(alu1+119)];
  var val152 = data2_65536[(alu1+12)];
  var val153 = data2_65536[(alu1+120)];
  var val154 = data2_65536[(alu1+121)];
  var val155 = data2_65536[(alu1+122)];
  var val156 = data2_65536[(alu1+123)];
  var val157 = data2_65536[(alu1+124)];
  var val158 = data2_65536[(alu1+125)];
  var val159 = data2_65536[(alu1+126)];
  var val160 = data2_65536[(alu1+127)];
  var val161 = data2_65536[(alu1+13)];
  var val162 = data2_65536[(alu1+14)];
  var val163 = data2_65536[(alu1+15)];
  var val164 = data2_65536[(alu1+16)];
  var val165 = data2_65536[(alu1+17)];
  var val166 = data2_65536[(alu1+18)];
  var val167 = data2_65536[(alu1+19)];
  var val168 = data2_65536[(alu1+2)];
  var val169 = data2_65536[(alu1+20)];
  var val170 = data2_65536[(alu1+21)];
  var val171 = data2_65536[(alu1+22)];
  var val172 = data2_65536[(alu1+23)];
  var val173 = data2_65536[(alu1+24)];
  var val174 = data2_65536[(alu1+25)];
  var val175 = data2_65536[(alu1+26)];
  var val176 = data2_65536[(alu1+27)];
  var val177 = data2_65536[(alu1+28)];
  var val178 = data2_65536[(alu1+29)];
  var val179 = data2_65536[(alu1+3)];
  var val180 = data2_65536[(alu1+30)];
  var val181 = data2_65536[(alu1+31)];
  var val182 = data2_65536[(alu1+32)];
  var val183 = data2_65536[(alu1+33)];
  var val184 = data2_65536[(alu1+34)];
  var val185 = data2_65536[(alu1+35)];
  var val186 = data2_65536[(alu1+36)];
  var val187 = data2_65536[(alu1+37)];
  var val188 = data2_65536[(alu1+38)];
  var val189 = data2_65536[(alu1+39)];
  var val190 = data2_65536[(alu1+4)];
  var val191 = data2_65536[(alu1+40)];
  var val192 = data2_65536[(alu1+41)];
  var val193 = data2_65536[(alu1+42)];
  var val194 = data2_65536[(alu1+43)];
  var val195 = data2_65536[(alu1+44)];
  var val196 = data2_65536[(alu1+45)];
  var val197 = data2_65536[(alu1+46)];
  var val198 = data2_65536[(alu1+47)];
  var val199 = data2_65536[(alu1+48)];
  var val200 = data2_65536[(alu1+49)];
  var val201 = data2_65536[(alu1+5)];
  var val202 = data2_65536[(alu1+50)];
  var val203 = data2_65536[(alu1+51)];
  var val204 = data2_65536[(alu1+52)];
  var val205 = data2_65536[(alu1+53)];
  var val206 = data2_65536[(alu1+54)];
  var val207 = data2_65536[(alu1+55)];
  var val208 = data2_65536[(alu1+56)];
  var val209 = data2_65536[(alu1+57)];
  var val210 = data2_65536[(alu1+58)];
  var val211 = data2_65536[(alu1+59)];
  var val212 = data2_65536[(alu1+6)];
  var val213 = data2_65536[(alu1+60)];
  var val214 = data2_65536[(alu1+61)];
  var val215 = data2_65536[(alu1+62)];
  var val216 = data2_65536[(alu1+63)];
  var val217 = data2_65536[(alu1+64)];
  var val218 = data2_65536[(alu1+65)];
  var val219 = data2_65536[(alu1+66)];
  var val220 = data2_65536[(alu1+67)];
  var val221 = data2_65536[(alu1+68)];
  var val222 = data2_65536[(alu1+69)];
  var val223 = data2_65536[(alu1+7)];
  var val224 = data2_65536[(alu1+70)];
  var val225 = data2_65536[(alu1+71)];
  var val226 = data2_65536[(alu1+72)];
  var val227 = data2_65536[(alu1+73)];
  var val228 = data2_65536[(alu1+74)];
  var val229 = data2_65536[(alu1+75)];
  var val230 = data2_65536[(alu1+76)];
  var val231 = data2_65536[(alu1+77)];
  var val232 = data2_65536[(alu1+78)];
  var val233 = data2_65536[(alu1+79)];
  var val234 = data2_65536[(alu1+8)];
  var val235 = data2_65536[(alu1+80)];
  var val236 = data2_65536[(alu1+81)];
  var val237 = data2_65536[(alu1+82)];
  var val238 = data2_65536[(alu1+83)];
  var val239 = data2_65536[(alu1+84)];
  var val240 = data2_65536[(alu1+85)];
  var val241 = data2_65536[(alu1+86)];
  var val242 = data2_65536[(alu1+87)];
  var val243 = data2_65536[(alu1+88)];
  var val244 = data2_65536[(alu1+89)];
  var val245 = data2_65536[(alu1+9)];
  var val246 = data2_65536[(alu1+90)];
  var val247 = data2_65536[(alu1+91)];
  var val248 = data2_65536[(alu1+92)];
  var val249 = data2_65536[(alu1+93)];
  var val250 = data2_65536[(alu1+94)];
  var val251 = data2_65536[(alu1+95)];
  var val252 = data2_65536[(alu1+96)];
  var val253 = data2_65536[(alu1+97)];
  var val254 = data2_65536[(alu1+98)];
  var val255 = data2_65536[(alu1+99)];
  var alu2 = (bitcast<i32>((cast4<<6u))+bitcast<i32>((cast5<<2u))+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast1<<16u)));
  data0_524288[alu2] = (((val0*val128)+(val1*val129)+(val40*val168)+(val51*val179)+(val62*val190)+(val73*val201)+(val84*val212)+(val95*val223)+(val106*val234)+(val117*val245)+(val2*val130)+(val13*val141)+(val24*val152)+(val33*val161)+(val34*val162)+(val35*val163)+(val36*val164)+(val37*val165)+(val38*val166)+(val39*val167)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val173)+(val46*val174)+(val47*val175)+(val48*val176)+(val49*val177)+(val50*val178)+(val52*val180)+(val53*val181))*0.1767766922712326f);
  data0_524288[(alu2+1)] = (((val0*val182)+(val1*val183)+(val40*val184)+(val51*val185)+(val62*val186)+(val73*val187)+(val84*val188)+(val95*val189)+(val106*val191)+(val117*val192)+(val2*val193)+(val13*val194)+(val24*val195)+(val33*val196)+(val34*val197)+(val35*val198)+(val36*val199)+(val37*val200)+(val38*val202)+(val39*val203)+(val41*val204)+(val42*val205)+(val43*val206)+(val44*val207)+(val45*val208)+(val46*val209)+(val47*val210)+(val48*val211)+(val49*val213)+(val50*val214)+(val52*val215)+(val53*val216))*0.1767766922712326f);
  data0_524288[(alu2+2)] = (((val0*val217)+(val1*val218)+(val40*val219)+(val51*val220)+(val62*val221)+(val73*val222)+(val84*val224)+(val95*val225)+(val106*val226)+(val117*val227)+(val2*val228)+(val13*val229)+(val24*val230)+(val33*val231)+(val34*val232)+(val35*val233)+(val36*val235)+(val37*val236)+(val38*val237)+(val39*val238)+(val41*val239)+(val42*val240)+(val43*val241)+(val44*val242)+(val45*val243)+(val46*val244)+(val47*val246)+(val48*val247)+(val49*val248)+(val50*val249)+(val52*val250)+(val53*val251))*0.1767766922712326f);
  data0_524288[(alu2+256)] = (((val54*val128)+(val55*val129)+(val56*val168)+(val57*val179)+(val58*val190)+(val59*val201)+(val60*val212)+(val61*val223)+(val63*val234)+(val64*val245)+(val65*val130)+(val66*val141)+(val67*val152)+(val68*val161)+(val69*val162)+(val70*val163)+(val71*val164)+(val72*val165)+(val74*val166)+(val75*val167)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val173)+(val81*val174)+(val82*val175)+(val83*val176)+(val85*val177)+(val86*val178)+(val87*val180)+(val88*val181))*0.1767766922712326f);
  data0_524288[(alu2+257)] = (((val54*val182)+(val55*val183)+(val56*val184)+(val57*val185)+(val58*val186)+(val59*val187)+(val60*val188)+(val61*val189)+(val63*val191)+(val64*val192)+(val65*val193)+(val66*val194)+(val67*val195)+(val68*val196)+(val69*val197)+(val70*val198)+(val71*val199)+(val72*val200)+(val74*val202)+(val75*val203)+(val76*val204)+(val77*val205)+(val78*val206)+(val79*val207)+(val80*val208)+(val81*val209)+(val82*val210)+(val83*val211)+(val85*val213)+(val86*val214)+(val87*val215)+(val88*val216))*0.1767766922712326f);
  data0_524288[(alu2+258)] = (((val54*val217)+(val55*val218)+(val56*val219)+(val57*val220)+(val58*val221)+(val59*val222)+(val60*val224)+(val61*val225)+(val63*val226)+(val64*val227)+(val65*val228)+(val66*val229)+(val67*val230)+(val68*val231)+(val69*val232)+(val70*val233)+(val71*val235)+(val72*val236)+(val74*val237)+(val75*val238)+(val76*val239)+(val77*val240)+(val78*val241)+(val79*val242)+(val80*val243)+(val81*val244)+(val82*val246)+(val83*val247)+(val85*val248)+(val86*val249)+(val87*val250)+(val88*val251))*0.1767766922712326f);
  data0_524288[(alu2+259)] = (((val54*val252)+(val55*val253)+(val56*val254)+(val57*val255)+(val58*val131)+(val59*val132)+(val60*val133)+(val61*val134)+(val63*val135)+(val64*val136)+(val65*val137)+(val66*val138)+(val67*val139)+(val68*val140)+(val69*val142)+(val70*val143)+(val71*val144)+(val72*val145)+(val74*val146)+(val75*val147)+(val76*val148)+(val77*val149)+(val78*val150)+(val79*val151)+(val80*val153)+(val81*val154)+(val82*val155)+(val83*val156)+(val85*val157)+(val86*val158)+(val87*val159)+(val88*val160))*0.1767766922712326f);
  data0_524288[(alu2+3)] = (((val0*val252)+(val1*val253)+(val40*val254)+(val51*val255)+(val62*val131)+(val73*val132)+(val84*val133)+(val95*val134)+(val106*val135)+(val117*val136)+(val2*val137)+(val13*val138)+(val24*val139)+(val33*val140)+(val34*val142)+(val35*val143)+(val36*val144)+(val37*val145)+(val38*val146)+(val39*val147)+(val41*val148)+(val42*val149)+(val43*val150)+(val44*val151)+(val45*val153)+(val46*val154)+(val47*val155)+(val48*val156)+(val49*val157)+(val50*val158)+(val52*val159)+(val53*val160))*0.1767766922712326f);
  data0_524288[(alu2+512)] = (((val89*val128)+(val90*val129)+(val91*val168)+(val92*val179)+(val93*val190)+(val94*val201)+(val96*val212)+(val97*val223)+(val98*val234)+(val99*val245)+(val100*val130)+(val101*val141)+(val102*val152)+(val103*val161)+(val104*val162)+(val105*val163)+(val107*val164)+(val108*val165)+(val109*val166)+(val110*val167)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val173)+(val116*val174)+(val118*val175)+(val119*val176)+(val120*val177)+(val121*val178)+(val122*val180)+(val123*val181))*0.1767766922712326f);
  data0_524288[(alu2+513)] = (((val89*val182)+(val90*val183)+(val91*val184)+(val92*val185)+(val93*val186)+(val94*val187)+(val96*val188)+(val97*val189)+(val98*val191)+(val99*val192)+(val100*val193)+(val101*val194)+(val102*val195)+(val103*val196)+(val104*val197)+(val105*val198)+(val107*val199)+(val108*val200)+(val109*val202)+(val110*val203)+(val111*val204)+(val112*val205)+(val113*val206)+(val114*val207)+(val115*val208)+(val116*val209)+(val118*val210)+(val119*val211)+(val120*val213)+(val121*val214)+(val122*val215)+(val123*val216))*0.1767766922712326f);
  data0_524288[(alu2+514)] = (((val89*val217)+(val90*val218)+(val91*val219)+(val92*val220)+(val93*val221)+(val94*val222)+(val96*val224)+(val97*val225)+(val98*val226)+(val99*val227)+(val100*val228)+(val101*val229)+(val102*val230)+(val103*val231)+(val104*val232)+(val105*val233)+(val107*val235)+(val108*val236)+(val109*val237)+(val110*val238)+(val111*val239)+(val112*val240)+(val113*val241)+(val114*val242)+(val115*val243)+(val116*val244)+(val118*val246)+(val119*val247)+(val120*val248)+(val121*val249)+(val122*val250)+(val123*val251))*0.1767766922712326f);
  data0_524288[(alu2+515)] = (((val89*val252)+(val90*val253)+(val91*val254)+(val92*val255)+(val93*val131)+(val94*val132)+(val96*val133)+(val97*val134)+(val98*val135)+(val99*val136)+(val100*val137)+(val101*val138)+(val102*val139)+(val103*val140)+(val104*val142)+(val105*val143)+(val107*val144)+(val108*val145)+(val109*val146)+(val110*val147)+(val111*val148)+(val112*val149)+(val113*val150)+(val114*val151)+(val115*val153)+(val116*val154)+(val118*val155)+(val119*val156)+(val120*val157)+(val121*val158)+(val122*val159)+(val123*val160))*0.1767766922712326f);
  data0_524288[(alu2+768)] = (((val124*val128)+(val125*val129)+(val126*val168)+(val127*val179)+(val3*val190)+(val4*val201)+(val5*val212)+(val6*val223)+(val7*val234)+(val8*val245)+(val9*val130)+(val10*val141)+(val11*val152)+(val12*val161)+(val14*val162)+(val15*val163)+(val16*val164)+(val17*val165)+(val18*val166)+(val19*val167)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val173)+(val26*val174)+(val27*val175)+(val28*val176)+(val29*val177)+(val30*val178)+(val31*val180)+(val32*val181))*0.1767766922712326f);
  data0_524288[(alu2+769)] = (((val124*val182)+(val125*val183)+(val126*val184)+(val127*val185)+(val3*val186)+(val4*val187)+(val5*val188)+(val6*val189)+(val7*val191)+(val8*val192)+(val9*val193)+(val10*val194)+(val11*val195)+(val12*val196)+(val14*val197)+(val15*val198)+(val16*val199)+(val17*val200)+(val18*val202)+(val19*val203)+(val20*val204)+(val21*val205)+(val22*val206)+(val23*val207)+(val25*val208)+(val26*val209)+(val27*val210)+(val28*val211)+(val29*val213)+(val30*val214)+(val31*val215)+(val32*val216))*0.1767766922712326f);
  data0_524288[(alu2+770)] = (((val124*val217)+(val125*val218)+(val126*val219)+(val127*val220)+(val3*val221)+(val4*val222)+(val5*val224)+(val6*val225)+(val7*val226)+(val8*val227)+(val9*val228)+(val10*val229)+(val11*val230)+(val12*val231)+(val14*val232)+(val15*val233)+(val16*val235)+(val17*val236)+(val18*val237)+(val19*val238)+(val20*val239)+(val21*val240)+(val22*val241)+(val23*val242)+(val25*val243)+(val26*val244)+(val27*val246)+(val28*val247)+(val29*val248)+(val30*val249)+(val31*val250)+(val32*val251))*0.1767766922712326f);
  data0_524288[(alu2+771)] = (((val124*val252)+(val125*val253)+(val126*val254)+(val127*val255)+(val3*val131)+(val4*val132)+(val5*val133)+(val6*val134)+(val7*val135)+(val8*val136)+(val9*val137)+(val10*val138)+(val11*val139)+(val12*val140)+(val14*val142)+(val15*val143)+(val16*val144)+(val17*val145)+(val18*val146)+(val19*val147)+(val20*val148)+(val21*val149)+(val22*val150)+(val23*val151)+(val25*val153)+(val26*val154)+(val27*val155)+(val28*val156)+(val29*val157)+(val30*val158)+(val31*val159)+(val32*val160))*0.1767766922712326f);
}`;

const r_2048_16_16_dispatch_133 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = (f32(-INFINITY));
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val0 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    var val1 = buf0[0];
    var alu1 = select(val1,val0,(val1<val0));
    buf0[0] = alu1;
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = (f32(-INFINITY));
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf1[0];
    var val3 = buf2[Lidx102];
    var alu7 = select(val2,val3,(val2<val3));
    buf1[0] = alu7;
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_2048_16_16_dispatch_134 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,16>;
@group(0) @binding(1)var<storage,read_write>data0_2048:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@compute @workgroup_size(16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,1>;
  var buf1: array<f32,1>;
  var gidx0 = i32(gindex.x); /* 2048 */
  var val0 = data2_2048[gidx0];
  var lidx0 = i32(lindex.x); /* 16 */
  buf0[0] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 16; Lidx0++) {
    var val1 = data1_524288[(bitcast<i32>((bitcast<u32>(lidx0)<<4u))+Lidx0+bitcast<i32>((bitcast<u32>(gidx0)<<8u)))];
    buf0[0] = (buf0[0]+exp2(((val1-val0)*1.4426950216293335f)));
  }
  buf2[lidx0] = buf0[0];
  workgroupBarrier();
  buf1[0] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 16; Lidx102++) {
    var val2 = buf2[Lidx102];
    buf1[0] = (buf1[0]+val2);
  }
  if ((lidx0==0)) {
    data0_2048[gidx0] = buf1[0];
  }
}`;

const r_8_4_16_8_4_4_64_4_dispatch_135 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_524288:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_2048:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_2048:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_196608:array<f32>;
@compute @workgroup_size(16,8) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast2<<2u))+bitcast<i32>((cast1<<8u)));
  var val0 = data2_2048[alu0];
  var alu1 = (alu0+1);
  var val1 = data2_2048[alu1];
  var alu2 = (alu0+2);
  var val2 = data2_2048[alu2];
  var alu3 = (alu0+3);
  var val3 = data2_2048[alu3];
  var lidx1 = i32(lindex.y); /* 8 */
  var cast3 = bitcast<i32>((bitcast<u32>(lidx1)<<2u));
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var alu20 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast2<<10u))+bitcast<i32>((bitcast<u32>(Lidx0)<<2u))+bitcast<i32>((cast1<<16u)));
    var val4 = data1_524288[alu20];
    var val5 = data1_524288[(alu20+1)];
    var val6 = data1_524288[(alu20+2)];
    var val7 = data1_524288[(alu20+256)];
    var val8 = data1_524288[(alu20+257)];
    var val9 = data1_524288[(alu20+258)];
    var val10 = data1_524288[(alu20+259)];
    var val11 = data1_524288[(alu20+3)];
    var val12 = data1_524288[(alu20+512)];
    var val13 = data1_524288[(alu20+513)];
    var val14 = data1_524288[(alu20+514)];
    var val15 = data1_524288[(alu20+515)];
    var val16 = data1_524288[(alu20+768)];
    var val17 = data1_524288[(alu20+769)];
    var val18 = data1_524288[(alu20+770)];
    var val19 = data1_524288[(alu20+771)];
    var alu21 = (bitcast<i32>((cast1<<5u))+cast3+(Lidx0*3072));
    var val20 = data4_196608[(alu21+1280)];
    var val21 = data4_196608[(alu21+1281)];
    var val22 = data4_196608[(alu21+1282)];
    var val23 = data4_196608[(alu21+1283)];
    var val24 = data4_196608[(alu21+2048)];
    var val25 = data4_196608[(alu21+2049)];
    var val26 = data4_196608[(alu21+2050)];
    var val27 = data4_196608[(alu21+2051)];
    var val28 = data4_196608[(alu21+2816)];
    var val29 = data4_196608[(alu21+2817)];
    var val30 = data4_196608[(alu21+2818)];
    var val31 = data4_196608[(alu21+2819)];
    var val32 = data4_196608[(alu21+512)];
    var val33 = data4_196608[(alu21+513)];
    var val34 = data4_196608[(alu21+514)];
    var val35 = data4_196608[(alu21+515)];
    var alu22 = exp2(((val4-val0)*1.4426950216293335f));
    var alu23 = exp2(((val5-val0)*1.4426950216293335f));
    var alu24 = exp2(((val6-val0)*1.4426950216293335f));
    var alu25 = exp2(((val7-val1)*1.4426950216293335f));
    var alu26 = exp2(((val8-val1)*1.4426950216293335f));
    var alu27 = exp2(((val9-val1)*1.4426950216293335f));
    var alu28 = exp2(((val10-val1)*1.4426950216293335f));
    var alu29 = exp2(((val11-val0)*1.4426950216293335f));
    var alu30 = exp2(((val12-val2)*1.4426950216293335f));
    var alu31 = exp2(((val13-val2)*1.4426950216293335f));
    var alu32 = exp2(((val14-val2)*1.4426950216293335f));
    var alu33 = exp2(((val15-val2)*1.4426950216293335f));
    var alu34 = exp2(((val16-val3)*1.4426950216293335f));
    var alu35 = exp2(((val17-val3)*1.4426950216293335f));
    var alu36 = exp2(((val18-val3)*1.4426950216293335f));
    var alu37 = exp2(((val19-val3)*1.4426950216293335f));
    buf0[0] = (buf0[0]+(alu22*val32)+(alu23*val20)+(alu24*val24)+(alu29*val28));
    buf0[1] = (buf0[1]+(alu22*val33)+(alu23*val21)+(alu24*val25)+(alu29*val29));
    buf0[10] = (buf0[10]+(alu30*val34)+(alu31*val22)+(alu32*val26)+(alu33*val30));
    buf0[11] = (buf0[11]+(alu30*val35)+(alu31*val23)+(alu32*val27)+(alu33*val31));
    buf0[12] = (buf0[12]+(alu34*val32)+(alu35*val20)+(alu36*val24)+(alu37*val28));
    buf0[13] = (buf0[13]+(alu34*val33)+(alu35*val21)+(alu36*val25)+(alu37*val29));
    buf0[14] = (buf0[14]+(alu34*val34)+(alu35*val22)+(alu36*val26)+(alu37*val30));
    buf0[15] = (buf0[15]+(alu34*val35)+(alu35*val23)+(alu36*val27)+(alu37*val31));
    buf0[2] = (buf0[2]+(alu22*val34)+(alu23*val22)+(alu24*val26)+(alu29*val30));
    buf0[3] = (buf0[3]+(alu22*val35)+(alu23*val23)+(alu24*val27)+(alu29*val31));
    buf0[4] = (buf0[4]+(alu25*val32)+(alu26*val20)+(alu27*val24)+(alu28*val28));
    buf0[5] = (buf0[5]+(alu25*val33)+(alu26*val21)+(alu27*val25)+(alu28*val29));
    buf0[6] = (buf0[6]+(alu25*val34)+(alu26*val22)+(alu27*val26)+(alu28*val30));
    buf0[7] = (buf0[7]+(alu25*val35)+(alu26*val23)+(alu27*val27)+(alu28*val31));
    buf0[8] = (buf0[8]+(alu30*val32)+(alu31*val20)+(alu32*val24)+(alu33*val28));
    buf0[9] = (buf0[9]+(alu30*val33)+(alu31*val21)+(alu32*val25)+(alu33*val29));
  }
  var val36 = data3_2048[alu0];
  var val37 = data3_2048[alu1];
  var val38 = data3_2048[alu2];
  var val39 = data3_2048[alu3];
  var alu55 = (bitcast<i32>((cast0<<11u))+bitcast<i32>((cast2<<7u))+cast3+bitcast<i32>((cast1<<13u)));
  var alu56 = (1/val36);
  var alu57 = (1/val37);
  var alu58 = (1/val38);
  var alu59 = (1/val39);
  data0_65536[alu55] = (buf0[0]*alu56);
  data0_65536[(alu55+1)] = (buf0[1]*alu56);
  data0_65536[(alu55+2)] = (buf0[2]*alu56);
  data0_65536[(alu55+3)] = (buf0[3]*alu56);
  data0_65536[(alu55+32)] = (buf0[4]*alu57);
  data0_65536[(alu55+33)] = (buf0[5]*alu57);
  data0_65536[(alu55+34)] = (buf0[6]*alu57);
  data0_65536[(alu55+35)] = (buf0[7]*alu57);
  data0_65536[(alu55+64)] = (buf0[8]*alu58);
  data0_65536[(alu55+65)] = (buf0[9]*alu58);
  data0_65536[(alu55+66)] = (buf0[10]*alu58);
  data0_65536[(alu55+67)] = (buf0[11]*alu58);
  data0_65536[(alu55+96)] = (buf0[12]*alu59);
  data0_65536[(alu55+97)] = (buf0[13]*alu59);
  data0_65536[(alu55+98)] = (buf0[14]*alu59);
  data0_65536[(alu55+99)] = (buf0[15]*alu59);
}`;

const r_8_4_8_16_4_4_8_32_dispatch_136 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_65536:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_65536:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx0);
  var cast3 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0_0 = 0; Lidx0_0 < 8; Lidx0_0++) {
    var cast4 = bitcast<u32>(Lidx0_0);
    var alu16 = (bitcast<i32>((cast1<<10u))+bitcast<i32>((cast2<<7u))+bitcast<i32>((cast4<<13u)));
    var val0 = data3_65536[alu16];
    var val1 = data3_65536[(alu16+1)];
    var val2 = data3_65536[(alu16+10)];
    var val3 = data3_65536[(alu16+100)];
    var val4 = data3_65536[(alu16+101)];
    var val5 = data3_65536[(alu16+102)];
    var val6 = data3_65536[(alu16+103)];
    var val7 = data3_65536[(alu16+104)];
    var val8 = data3_65536[(alu16+105)];
    var val9 = data3_65536[(alu16+106)];
    var val10 = data3_65536[(alu16+107)];
    var val11 = data3_65536[(alu16+108)];
    var val12 = data3_65536[(alu16+109)];
    var val13 = data3_65536[(alu16+11)];
    var val14 = data3_65536[(alu16+110)];
    var val15 = data3_65536[(alu16+111)];
    var val16 = data3_65536[(alu16+112)];
    var val17 = data3_65536[(alu16+113)];
    var val18 = data3_65536[(alu16+114)];
    var val19 = data3_65536[(alu16+115)];
    var val20 = data3_65536[(alu16+116)];
    var val21 = data3_65536[(alu16+117)];
    var val22 = data3_65536[(alu16+118)];
    var val23 = data3_65536[(alu16+119)];
    var val24 = data3_65536[(alu16+12)];
    var val25 = data3_65536[(alu16+120)];
    var val26 = data3_65536[(alu16+121)];
    var val27 = data3_65536[(alu16+122)];
    var val28 = data3_65536[(alu16+123)];
    var val29 = data3_65536[(alu16+124)];
    var val30 = data3_65536[(alu16+125)];
    var val31 = data3_65536[(alu16+126)];
    var val32 = data3_65536[(alu16+127)];
    var val33 = data3_65536[(alu16+13)];
    var val34 = data3_65536[(alu16+14)];
    var val35 = data3_65536[(alu16+15)];
    var val36 = data3_65536[(alu16+16)];
    var val37 = data3_65536[(alu16+17)];
    var val38 = data3_65536[(alu16+18)];
    var val39 = data3_65536[(alu16+19)];
    var val40 = data3_65536[(alu16+2)];
    var val41 = data3_65536[(alu16+20)];
    var val42 = data3_65536[(alu16+21)];
    var val43 = data3_65536[(alu16+22)];
    var val44 = data3_65536[(alu16+23)];
    var val45 = data3_65536[(alu16+24)];
    var val46 = data3_65536[(alu16+25)];
    var val47 = data3_65536[(alu16+26)];
    var val48 = data3_65536[(alu16+27)];
    var val49 = data3_65536[(alu16+28)];
    var val50 = data3_65536[(alu16+29)];
    var val51 = data3_65536[(alu16+3)];
    var val52 = data3_65536[(alu16+30)];
    var val53 = data3_65536[(alu16+31)];
    var val54 = data3_65536[(alu16+32)];
    var val55 = data3_65536[(alu16+33)];
    var val56 = data3_65536[(alu16+34)];
    var val57 = data3_65536[(alu16+35)];
    var val58 = data3_65536[(alu16+36)];
    var val59 = data3_65536[(alu16+37)];
    var val60 = data3_65536[(alu16+38)];
    var val61 = data3_65536[(alu16+39)];
    var val62 = data3_65536[(alu16+4)];
    var val63 = data3_65536[(alu16+40)];
    var val64 = data3_65536[(alu16+41)];
    var val65 = data3_65536[(alu16+42)];
    var val66 = data3_65536[(alu16+43)];
    var val67 = data3_65536[(alu16+44)];
    var val68 = data3_65536[(alu16+45)];
    var val69 = data3_65536[(alu16+46)];
    var val70 = data3_65536[(alu16+47)];
    var val71 = data3_65536[(alu16+48)];
    var val72 = data3_65536[(alu16+49)];
    var val73 = data3_65536[(alu16+5)];
    var val74 = data3_65536[(alu16+50)];
    var val75 = data3_65536[(alu16+51)];
    var val76 = data3_65536[(alu16+52)];
    var val77 = data3_65536[(alu16+53)];
    var val78 = data3_65536[(alu16+54)];
    var val79 = data3_65536[(alu16+55)];
    var val80 = data3_65536[(alu16+56)];
    var val81 = data3_65536[(alu16+57)];
    var val82 = data3_65536[(alu16+58)];
    var val83 = data3_65536[(alu16+59)];
    var val84 = data3_65536[(alu16+6)];
    var val85 = data3_65536[(alu16+60)];
    var val86 = data3_65536[(alu16+61)];
    var val87 = data3_65536[(alu16+62)];
    var val88 = data3_65536[(alu16+63)];
    var val89 = data3_65536[(alu16+64)];
    var val90 = data3_65536[(alu16+65)];
    var val91 = data3_65536[(alu16+66)];
    var val92 = data3_65536[(alu16+67)];
    var val93 = data3_65536[(alu16+68)];
    var val94 = data3_65536[(alu16+69)];
    var val95 = data3_65536[(alu16+7)];
    var val96 = data3_65536[(alu16+70)];
    var val97 = data3_65536[(alu16+71)];
    var val98 = data3_65536[(alu16+72)];
    var val99 = data3_65536[(alu16+73)];
    var val100 = data3_65536[(alu16+74)];
    var val101 = data3_65536[(alu16+75)];
    var val102 = data3_65536[(alu16+76)];
    var val103 = data3_65536[(alu16+77)];
    var val104 = data3_65536[(alu16+78)];
    var val105 = data3_65536[(alu16+79)];
    var val106 = data3_65536[(alu16+8)];
    var val107 = data3_65536[(alu16+80)];
    var val108 = data3_65536[(alu16+81)];
    var val109 = data3_65536[(alu16+82)];
    var val110 = data3_65536[(alu16+83)];
    var val111 = data3_65536[(alu16+84)];
    var val112 = data3_65536[(alu16+85)];
    var val113 = data3_65536[(alu16+86)];
    var val114 = data3_65536[(alu16+87)];
    var val115 = data3_65536[(alu16+88)];
    var val116 = data3_65536[(alu16+89)];
    var val117 = data3_65536[(alu16+9)];
    var val118 = data3_65536[(alu16+90)];
    var val119 = data3_65536[(alu16+91)];
    var val120 = data3_65536[(alu16+92)];
    var val121 = data3_65536[(alu16+93)];
    var val122 = data3_65536[(alu16+94)];
    var val123 = data3_65536[(alu16+95)];
    var val124 = data3_65536[(alu16+96)];
    var val125 = data3_65536[(alu16+97)];
    var val126 = data3_65536[(alu16+98)];
    var val127 = data3_65536[(alu16+99)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast3<<10u))+bitcast<i32>((cast4<<5u)));
    var val128 = data4_65536[alu17];
    var val129 = data4_65536[(alu17+1)];
    var val130 = data4_65536[(alu17+10)];
    var val131 = data4_65536[(alu17+11)];
    var val132 = data4_65536[(alu17+12)];
    var val133 = data4_65536[(alu17+13)];
    var val134 = data4_65536[(alu17+14)];
    var val135 = data4_65536[(alu17+15)];
    var val136 = data4_65536[(alu17+16)];
    var val137 = data4_65536[(alu17+17)];
    var val138 = data4_65536[(alu17+18)];
    var val139 = data4_65536[(alu17+19)];
    var val140 = data4_65536[(alu17+2)];
    var val141 = data4_65536[(alu17+20)];
    var val142 = data4_65536[(alu17+21)];
    var val143 = data4_65536[(alu17+22)];
    var val144 = data4_65536[(alu17+23)];
    var val145 = data4_65536[(alu17+24)];
    var val146 = data4_65536[(alu17+25)];
    var val147 = data4_65536[(alu17+256)];
    var val148 = data4_65536[(alu17+257)];
    var val149 = data4_65536[(alu17+258)];
    var val150 = data4_65536[(alu17+259)];
    var val151 = data4_65536[(alu17+26)];
    var val152 = data4_65536[(alu17+260)];
    var val153 = data4_65536[(alu17+261)];
    var val154 = data4_65536[(alu17+262)];
    var val155 = data4_65536[(alu17+263)];
    var val156 = data4_65536[(alu17+264)];
    var val157 = data4_65536[(alu17+265)];
    var val158 = data4_65536[(alu17+266)];
    var val159 = data4_65536[(alu17+267)];
    var val160 = data4_65536[(alu17+268)];
    var val161 = data4_65536[(alu17+269)];
    var val162 = data4_65536[(alu17+27)];
    var val163 = data4_65536[(alu17+270)];
    var val164 = data4_65536[(alu17+271)];
    var val165 = data4_65536[(alu17+272)];
    var val166 = data4_65536[(alu17+273)];
    var val167 = data4_65536[(alu17+274)];
    var val168 = data4_65536[(alu17+275)];
    var val169 = data4_65536[(alu17+276)];
    var val170 = data4_65536[(alu17+277)];
    var val171 = data4_65536[(alu17+278)];
    var val172 = data4_65536[(alu17+279)];
    var val173 = data4_65536[(alu17+28)];
    var val174 = data4_65536[(alu17+280)];
    var val175 = data4_65536[(alu17+281)];
    var val176 = data4_65536[(alu17+282)];
    var val177 = data4_65536[(alu17+283)];
    var val178 = data4_65536[(alu17+284)];
    var val179 = data4_65536[(alu17+285)];
    var val180 = data4_65536[(alu17+286)];
    var val181 = data4_65536[(alu17+287)];
    var val182 = data4_65536[(alu17+29)];
    var val183 = data4_65536[(alu17+3)];
    var val184 = data4_65536[(alu17+30)];
    var val185 = data4_65536[(alu17+31)];
    var val186 = data4_65536[(alu17+4)];
    var val187 = data4_65536[(alu17+5)];
    var val188 = data4_65536[(alu17+512)];
    var val189 = data4_65536[(alu17+513)];
    var val190 = data4_65536[(alu17+514)];
    var val191 = data4_65536[(alu17+515)];
    var val192 = data4_65536[(alu17+516)];
    var val193 = data4_65536[(alu17+517)];
    var val194 = data4_65536[(alu17+518)];
    var val195 = data4_65536[(alu17+519)];
    var val196 = data4_65536[(alu17+520)];
    var val197 = data4_65536[(alu17+521)];
    var val198 = data4_65536[(alu17+522)];
    var val199 = data4_65536[(alu17+523)];
    var val200 = data4_65536[(alu17+524)];
    var val201 = data4_65536[(alu17+525)];
    var val202 = data4_65536[(alu17+526)];
    var val203 = data4_65536[(alu17+527)];
    var val204 = data4_65536[(alu17+528)];
    var val205 = data4_65536[(alu17+529)];
    var val206 = data4_65536[(alu17+530)];
    var val207 = data4_65536[(alu17+531)];
    var val208 = data4_65536[(alu17+532)];
    var val209 = data4_65536[(alu17+533)];
    var val210 = data4_65536[(alu17+534)];
    var val211 = data4_65536[(alu17+535)];
    var val212 = data4_65536[(alu17+536)];
    var val213 = data4_65536[(alu17+537)];
    var val214 = data4_65536[(alu17+538)];
    var val215 = data4_65536[(alu17+539)];
    var val216 = data4_65536[(alu17+540)];
    var val217 = data4_65536[(alu17+541)];
    var val218 = data4_65536[(alu17+542)];
    var val219 = data4_65536[(alu17+543)];
    var val220 = data4_65536[(alu17+6)];
    var val221 = data4_65536[(alu17+7)];
    var val222 = data4_65536[(alu17+768)];
    var val223 = data4_65536[(alu17+769)];
    var val224 = data4_65536[(alu17+770)];
    var val225 = data4_65536[(alu17+771)];
    var val226 = data4_65536[(alu17+772)];
    var val227 = data4_65536[(alu17+773)];
    var val228 = data4_65536[(alu17+774)];
    var val229 = data4_65536[(alu17+775)];
    var val230 = data4_65536[(alu17+776)];
    var val231 = data4_65536[(alu17+777)];
    var val232 = data4_65536[(alu17+778)];
    var val233 = data4_65536[(alu17+779)];
    var val234 = data4_65536[(alu17+780)];
    var val235 = data4_65536[(alu17+781)];
    var val236 = data4_65536[(alu17+782)];
    var val237 = data4_65536[(alu17+783)];
    var val238 = data4_65536[(alu17+784)];
    var val239 = data4_65536[(alu17+785)];
    var val240 = data4_65536[(alu17+786)];
    var val241 = data4_65536[(alu17+787)];
    var val242 = data4_65536[(alu17+788)];
    var val243 = data4_65536[(alu17+789)];
    var val244 = data4_65536[(alu17+790)];
    var val245 = data4_65536[(alu17+791)];
    var val246 = data4_65536[(alu17+792)];
    var val247 = data4_65536[(alu17+793)];
    var val248 = data4_65536[(alu17+794)];
    var val249 = data4_65536[(alu17+795)];
    var val250 = data4_65536[(alu17+796)];
    var val251 = data4_65536[(alu17+797)];
    var val252 = data4_65536[(alu17+798)];
    var val253 = data4_65536[(alu17+799)];
    var val254 = data4_65536[(alu17+8)];
    var val255 = data4_65536[(alu17+9)];
    buf0[0] = (buf0[0]+(val0*val128)+(val1*val129)+(val40*val140)+(val51*val183)+(val62*val186)+(val73*val187)+(val84*val220)+(val95*val221)+(val106*val254)+(val117*val255)+(val2*val130)+(val13*val131)+(val24*val132)+(val33*val133)+(val34*val134)+(val35*val135)+(val36*val136)+(val37*val137)+(val38*val138)+(val39*val139)+(val41*val141)+(val42*val142)+(val43*val143)+(val44*val144)+(val45*val145)+(val46*val146)+(val47*val151)+(val48*val162)+(val49*val173)+(val50*val182)+(val52*val184)+(val53*val185));
    buf0[1] = (buf0[1]+(val0*val147)+(val1*val148)+(val40*val149)+(val51*val150)+(val62*val152)+(val73*val153)+(val84*val154)+(val95*val155)+(val106*val156)+(val117*val157)+(val2*val158)+(val13*val159)+(val24*val160)+(val33*val161)+(val34*val163)+(val35*val164)+(val36*val165)+(val37*val166)+(val38*val167)+(val39*val168)+(val41*val169)+(val42*val170)+(val43*val171)+(val44*val172)+(val45*val174)+(val46*val175)+(val47*val176)+(val48*val177)+(val49*val178)+(val50*val179)+(val52*val180)+(val53*val181));
    buf0[10] = (buf0[10]+(val89*val188)+(val90*val189)+(val91*val190)+(val92*val191)+(val93*val192)+(val94*val193)+(val96*val194)+(val97*val195)+(val98*val196)+(val99*val197)+(val100*val198)+(val101*val199)+(val102*val200)+(val103*val201)+(val104*val202)+(val105*val203)+(val107*val204)+(val108*val205)+(val109*val206)+(val110*val207)+(val111*val208)+(val112*val209)+(val113*val210)+(val114*val211)+(val115*val212)+(val116*val213)+(val118*val214)+(val119*val215)+(val120*val216)+(val121*val217)+(val122*val218)+(val123*val219));
    buf0[11] = (buf0[11]+(val89*val222)+(val90*val223)+(val91*val224)+(val92*val225)+(val93*val226)+(val94*val227)+(val96*val228)+(val97*val229)+(val98*val230)+(val99*val231)+(val100*val232)+(val101*val233)+(val102*val234)+(val103*val235)+(val104*val236)+(val105*val237)+(val107*val238)+(val108*val239)+(val109*val240)+(val110*val241)+(val111*val242)+(val112*val243)+(val113*val244)+(val114*val245)+(val115*val246)+(val116*val247)+(val118*val248)+(val119*val249)+(val120*val250)+(val121*val251)+(val122*val252)+(val123*val253));
    buf0[12] = (buf0[12]+(val124*val128)+(val125*val129)+(val126*val140)+(val127*val183)+(val3*val186)+(val4*val187)+(val5*val220)+(val6*val221)+(val7*val254)+(val8*val255)+(val9*val130)+(val10*val131)+(val11*val132)+(val12*val133)+(val14*val134)+(val15*val135)+(val16*val136)+(val17*val137)+(val18*val138)+(val19*val139)+(val20*val141)+(val21*val142)+(val22*val143)+(val23*val144)+(val25*val145)+(val26*val146)+(val27*val151)+(val28*val162)+(val29*val173)+(val30*val182)+(val31*val184)+(val32*val185));
    buf0[13] = (buf0[13]+(val124*val147)+(val125*val148)+(val126*val149)+(val127*val150)+(val3*val152)+(val4*val153)+(val5*val154)+(val6*val155)+(val7*val156)+(val8*val157)+(val9*val158)+(val10*val159)+(val11*val160)+(val12*val161)+(val14*val163)+(val15*val164)+(val16*val165)+(val17*val166)+(val18*val167)+(val19*val168)+(val20*val169)+(val21*val170)+(val22*val171)+(val23*val172)+(val25*val174)+(val26*val175)+(val27*val176)+(val28*val177)+(val29*val178)+(val30*val179)+(val31*val180)+(val32*val181));
    buf0[14] = (buf0[14]+(val124*val188)+(val125*val189)+(val126*val190)+(val127*val191)+(val3*val192)+(val4*val193)+(val5*val194)+(val6*val195)+(val7*val196)+(val8*val197)+(val9*val198)+(val10*val199)+(val11*val200)+(val12*val201)+(val14*val202)+(val15*val203)+(val16*val204)+(val17*val205)+(val18*val206)+(val19*val207)+(val20*val208)+(val21*val209)+(val22*val210)+(val23*val211)+(val25*val212)+(val26*val213)+(val27*val214)+(val28*val215)+(val29*val216)+(val30*val217)+(val31*val218)+(val32*val219));
    buf0[15] = (buf0[15]+(val124*val222)+(val125*val223)+(val126*val224)+(val127*val225)+(val3*val226)+(val4*val227)+(val5*val228)+(val6*val229)+(val7*val230)+(val8*val231)+(val9*val232)+(val10*val233)+(val11*val234)+(val12*val235)+(val14*val236)+(val15*val237)+(val16*val238)+(val17*val239)+(val18*val240)+(val19*val241)+(val20*val242)+(val21*val243)+(val22*val244)+(val23*val245)+(val25*val246)+(val26*val247)+(val27*val248)+(val28*val249)+(val29*val250)+(val30*val251)+(val31*val252)+(val32*val253));
    buf0[2] = (buf0[2]+(val0*val188)+(val1*val189)+(val40*val190)+(val51*val191)+(val62*val192)+(val73*val193)+(val84*val194)+(val95*val195)+(val106*val196)+(val117*val197)+(val2*val198)+(val13*val199)+(val24*val200)+(val33*val201)+(val34*val202)+(val35*val203)+(val36*val204)+(val37*val205)+(val38*val206)+(val39*val207)+(val41*val208)+(val42*val209)+(val43*val210)+(val44*val211)+(val45*val212)+(val46*val213)+(val47*val214)+(val48*val215)+(val49*val216)+(val50*val217)+(val52*val218)+(val53*val219));
    buf0[3] = (buf0[3]+(val0*val222)+(val1*val223)+(val40*val224)+(val51*val225)+(val62*val226)+(val73*val227)+(val84*val228)+(val95*val229)+(val106*val230)+(val117*val231)+(val2*val232)+(val13*val233)+(val24*val234)+(val33*val235)+(val34*val236)+(val35*val237)+(val36*val238)+(val37*val239)+(val38*val240)+(val39*val241)+(val41*val242)+(val42*val243)+(val43*val244)+(val44*val245)+(val45*val246)+(val46*val247)+(val47*val248)+(val48*val249)+(val49*val250)+(val50*val251)+(val52*val252)+(val53*val253));
    buf0[4] = (buf0[4]+(val54*val128)+(val55*val129)+(val56*val140)+(val57*val183)+(val58*val186)+(val59*val187)+(val60*val220)+(val61*val221)+(val63*val254)+(val64*val255)+(val65*val130)+(val66*val131)+(val67*val132)+(val68*val133)+(val69*val134)+(val70*val135)+(val71*val136)+(val72*val137)+(val74*val138)+(val75*val139)+(val76*val141)+(val77*val142)+(val78*val143)+(val79*val144)+(val80*val145)+(val81*val146)+(val82*val151)+(val83*val162)+(val85*val173)+(val86*val182)+(val87*val184)+(val88*val185));
    buf0[5] = (buf0[5]+(val54*val147)+(val55*val148)+(val56*val149)+(val57*val150)+(val58*val152)+(val59*val153)+(val60*val154)+(val61*val155)+(val63*val156)+(val64*val157)+(val65*val158)+(val66*val159)+(val67*val160)+(val68*val161)+(val69*val163)+(val70*val164)+(val71*val165)+(val72*val166)+(val74*val167)+(val75*val168)+(val76*val169)+(val77*val170)+(val78*val171)+(val79*val172)+(val80*val174)+(val81*val175)+(val82*val176)+(val83*val177)+(val85*val178)+(val86*val179)+(val87*val180)+(val88*val181));
    buf0[6] = (buf0[6]+(val54*val188)+(val55*val189)+(val56*val190)+(val57*val191)+(val58*val192)+(val59*val193)+(val60*val194)+(val61*val195)+(val63*val196)+(val64*val197)+(val65*val198)+(val66*val199)+(val67*val200)+(val68*val201)+(val69*val202)+(val70*val203)+(val71*val204)+(val72*val205)+(val74*val206)+(val75*val207)+(val76*val208)+(val77*val209)+(val78*val210)+(val79*val211)+(val80*val212)+(val81*val213)+(val82*val214)+(val83*val215)+(val85*val216)+(val86*val217)+(val87*val218)+(val88*val219));
    buf0[7] = (buf0[7]+(val54*val222)+(val55*val223)+(val56*val224)+(val57*val225)+(val58*val226)+(val59*val227)+(val60*val228)+(val61*val229)+(val63*val230)+(val64*val231)+(val65*val232)+(val66*val233)+(val67*val234)+(val68*val235)+(val69*val236)+(val70*val237)+(val71*val238)+(val72*val239)+(val74*val240)+(val75*val241)+(val76*val242)+(val77*val243)+(val78*val244)+(val79*val245)+(val80*val246)+(val81*val247)+(val82*val248)+(val83*val249)+(val85*val250)+(val86*val251)+(val87*val252)+(val88*val253));
    buf0[8] = (buf0[8]+(val89*val128)+(val90*val129)+(val91*val140)+(val92*val183)+(val93*val186)+(val94*val187)+(val96*val220)+(val97*val221)+(val98*val254)+(val99*val255)+(val100*val130)+(val101*val131)+(val102*val132)+(val103*val133)+(val104*val134)+(val105*val135)+(val107*val136)+(val108*val137)+(val109*val138)+(val110*val139)+(val111*val141)+(val112*val142)+(val113*val143)+(val114*val144)+(val115*val145)+(val116*val146)+(val118*val151)+(val119*val162)+(val120*val173)+(val121*val182)+(val122*val184)+(val123*val185));
    buf0[9] = (buf0[9]+(val89*val147)+(val90*val148)+(val91*val149)+(val92*val150)+(val93*val152)+(val94*val153)+(val96*val154)+(val97*val155)+(val98*val156)+(val99*val157)+(val100*val158)+(val101*val159)+(val102*val160)+(val103*val161)+(val104*val163)+(val105*val164)+(val107*val165)+(val108*val166)+(val109*val167)+(val110*val168)+(val111*val169)+(val112*val170)+(val113*val171)+(val114*val172)+(val115*val174)+(val116*val175)+(val118*val176)+(val119*val177)+(val120*val178)+(val121*val179)+(val122*val180)+(val123*val181));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast3<<2u)));
  var alu36 = (alu35+bitcast<i32>((cast1<<13u))+bitcast<i32>((cast2<<10u)));
  var val256 = data1_65536[alu36];
  var alu37 = (alu36+1);
  var val257 = data1_65536[alu37];
  var alu38 = (alu36+2);
  var val258 = data1_65536[alu38];
  var alu39 = (alu36+256);
  var val259 = data1_65536[alu39];
  var alu40 = (alu36+257);
  var val260 = data1_65536[alu40];
  var alu41 = (alu36+258);
  var val261 = data1_65536[alu41];
  var alu42 = (alu36+259);
  var val262 = data1_65536[alu42];
  var alu43 = (alu36+3);
  var val263 = data1_65536[alu43];
  var alu44 = (alu36+512);
  var val264 = data1_65536[alu44];
  var alu45 = (alu36+513);
  var val265 = data1_65536[alu45];
  var alu46 = (alu36+514);
  var val266 = data1_65536[alu46];
  var alu47 = (alu36+515);
  var val267 = data1_65536[alu47];
  var alu48 = (alu36+768);
  var val268 = data1_65536[alu48];
  var alu49 = (alu36+769);
  var val269 = data1_65536[alu49];
  var alu50 = (alu36+770);
  var val270 = data1_65536[alu50];
  var alu51 = (alu36+771);
  var val271 = data1_65536[alu51];
  var val272 = data2_1536[(alu35+512)];
  var val273 = data2_1536[(alu35+513)];
  var val274 = data2_1536[(alu35+514)];
  var val275 = data2_1536[(alu35+515)];
  var val276 = data5_256[alu35];
  var val277 = data5_256[(alu35+1)];
  var val278 = data5_256[(alu35+2)];
  var val279 = data5_256[(alu35+3)];
  data0_65536[alu36] = (val256+(val272*(buf0[0]+val276)));
  data0_65536[alu37] = (val257+(val273*(buf0[1]+val277)));
  data0_65536[alu38] = (val258+(val274*(buf0[2]+val278)));
  data0_65536[alu39] = (val259+(val272*(buf0[4]+val276)));
  data0_65536[alu40] = (val260+(val273*(buf0[5]+val277)));
  data0_65536[alu41] = (val261+(val274*(buf0[6]+val278)));
  data0_65536[alu42] = (val262+(val275*(buf0[7]+val279)));
  data0_65536[alu43] = (val263+(val275*(buf0[3]+val279)));
  data0_65536[alu44] = (val264+(val272*(buf0[8]+val276)));
  data0_65536[alu45] = (val265+(val273*(buf0[9]+val277)));
  data0_65536[alu46] = (val266+(val274*(buf0[10]+val278)));
  data0_65536[alu47] = (val267+(val275*(buf0[11]+val279)));
  data0_65536[alu48] = (val268+(val272*(buf0[12]+val276)));
  data0_65536[alu49] = (val269+(val273*(buf0[13]+val277)));
  data0_65536[alu50] = (val270+(val274*(buf0[14]+val278)));
  data0_65536[alu51] = (val271+(val275*(buf0[15]+val279)));
}`;

const r_16_8_4_4_32_dispatch_137 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_138 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_1536:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var val21 = data3_256[(alu0+1)];
  var val22 = data3_256[(alu0+2)];
  var val23 = data3_256[(alu0+3)];
  var val24 = data4_1536[(alu0+1024)];
  var val25 = data4_1536[(alu0+1025)];
  var val26 = data4_1536[(alu0+1026)];
  var val27 = data4_1536[(alu0+1027)];
  var val28 = data4_1536[(alu0+768)];
  var val29 = data4_1536[(alu0+769)];
  var val30 = data4_1536[(alu0+770)];
  var val31 = data4_1536[(alu0+771)];
  var alu18 = (1/val16);
  var alu19 = (1/val17);
  var alu20 = (1/val18);
  var alu21 = (1/val19);
  var alu22 = (1.0f+val24);
  var alu23 = (1.0f+val25);
  var alu24 = (1.0f+val26);
  var alu25 = (1.0f+val27);
  data0_65536[alu1] = ((val0*alu18*val20*alu22)+val28);
  data0_65536[alu2] = ((val1*alu18*val21*alu23)+val29);
  data0_65536[alu3] = ((val2*alu18*val22*alu24)+val30);
  data0_65536[alu4] = ((val3*alu19*val20*alu22)+val28);
  data0_65536[alu5] = ((val4*alu19*val21*alu23)+val29);
  data0_65536[alu6] = ((val5*alu19*val22*alu24)+val30);
  data0_65536[alu7] = ((val6*alu19*val23*alu25)+val31);
  data0_65536[alu8] = ((val7*alu18*val23*alu25)+val31);
  data0_65536[alu9] = ((val8*alu20*val20*alu22)+val28);
  data0_65536[alu10] = ((val9*alu20*val21*alu23)+val29);
  data0_65536[alu11] = ((val10*alu20*val22*alu24)+val30);
  data0_65536[alu12] = ((val11*alu20*val23*alu25)+val31);
  data0_65536[alu13] = ((val12*alu21*val20*alu22)+val28);
  data0_65536[alu14] = ((val13*alu21*val21*alu23)+val29);
  data0_65536[alu15] = ((val14*alu21*val22*alu24)+val30);
  data0_65536[alu16] = ((val15*alu21*val23*alu25)+val31);
}`;

const r_8_22_8_16_4_4_64_4_dispatch_139 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_360448:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_360448:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_1408:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 22 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast2 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = (bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u))+cast2);
    var val0 = data1_65536[alu16];
    var val1 = data1_65536[(alu16+1)];
    var val2 = data1_65536[(alu16+2)];
    var val3 = data1_65536[(alu16+256)];
    var val4 = data1_65536[(alu16+257)];
    var val5 = data1_65536[(alu16+258)];
    var val6 = data1_65536[(alu16+259)];
    var val7 = data1_65536[(alu16+3)];
    var val8 = data1_65536[(alu16+512)];
    var val9 = data1_65536[(alu16+513)];
    var val10 = data1_65536[(alu16+514)];
    var val11 = data1_65536[(alu16+515)];
    var val12 = data1_65536[(alu16+768)];
    var val13 = data1_65536[(alu16+769)];
    var val14 = data1_65536[(alu16+770)];
    var val15 = data1_65536[(alu16+771)];
    var alu17 = (bitcast<i32>((cast0<<14u))+bitcast<i32>((cast1<<10u))+cast2);
    var val16 = data2_360448[alu17];
    var val17 = data2_360448[(alu17+1)];
    var val18 = data2_360448[(alu17+2)];
    var val19 = data2_360448[(alu17+256)];
    var val20 = data2_360448[(alu17+257)];
    var val21 = data2_360448[(alu17+258)];
    var val22 = data2_360448[(alu17+259)];
    var val23 = data2_360448[(alu17+3)];
    var val24 = data2_360448[(alu17+512)];
    var val25 = data2_360448[(alu17+513)];
    var val26 = data2_360448[(alu17+514)];
    var val27 = data2_360448[(alu17+515)];
    var val28 = data2_360448[(alu17+768)];
    var val29 = data2_360448[(alu17+769)];
    var val30 = data2_360448[(alu17+770)];
    var val31 = data2_360448[(alu17+771)];
    buf0[0] = (buf0[0]+(val0*val16)+(val1*val17)+(val2*val18)+(val7*val23));
    buf0[1] = (buf0[1]+(val0*val19)+(val1*val20)+(val2*val21)+(val7*val22));
    buf0[10] = (buf0[10]+(val8*val24)+(val9*val25)+(val10*val26)+(val11*val27));
    buf0[11] = (buf0[11]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
    buf0[12] = (buf0[12]+(val12*val16)+(val13*val17)+(val14*val18)+(val15*val23));
    buf0[13] = (buf0[13]+(val12*val19)+(val13*val20)+(val14*val21)+(val15*val22));
    buf0[14] = (buf0[14]+(val12*val24)+(val13*val25)+(val14*val26)+(val15*val27));
    buf0[15] = (buf0[15]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    buf0[2] = (buf0[2]+(val0*val24)+(val1*val25)+(val2*val26)+(val7*val27));
    buf0[3] = (buf0[3]+(val0*val28)+(val1*val29)+(val2*val30)+(val7*val31));
    buf0[4] = (buf0[4]+(val3*val16)+(val4*val17)+(val5*val18)+(val6*val23));
    buf0[5] = (buf0[5]+(val3*val19)+(val4*val20)+(val5*val21)+(val6*val22));
    buf0[6] = (buf0[6]+(val3*val24)+(val4*val25)+(val5*val26)+(val6*val27));
    buf0[7] = (buf0[7]+(val3*val28)+(val4*val29)+(val5*val30)+(val6*val31));
    buf0[8] = (buf0[8]+(val8*val16)+(val9*val17)+(val10*val18)+(val11*val23));
    buf0[9] = (buf0[9]+(val8*val19)+(val9*val20)+(val10*val21)+(val11*val22));
  }
  var alu35 = (bitcast<i32>((cast0<<6u))+bitcast<i32>((cast1<<2u)));
  var val32 = data3_1408[alu35];
  var val33 = data3_1408[(alu35+1)];
  var val34 = data3_1408[(alu35+2)];
  var val35 = data3_1408[(alu35+3)];
  var alu36 = (alu35+(gidx1*45056)+(lidx0*5632));
  data0_360448[alu36] = (buf0[0]+val32);
  data0_360448[(alu36+1)] = (buf0[1]+val33);
  data0_360448[(alu36+1408)] = (buf0[4]+val32);
  data0_360448[(alu36+1409)] = (buf0[5]+val33);
  data0_360448[(alu36+1410)] = (buf0[6]+val34);
  data0_360448[(alu36+1411)] = (buf0[7]+val35);
  data0_360448[(alu36+2)] = (buf0[2]+val34);
  data0_360448[(alu36+2816)] = (buf0[8]+val32);
  data0_360448[(alu36+2817)] = (buf0[9]+val33);
  data0_360448[(alu36+2818)] = (buf0[10]+val34);
  data0_360448[(alu36+2819)] = (buf0[11]+val35);
  data0_360448[(alu36+3)] = (buf0[3]+val35);
  data0_360448[(alu36+4224)] = (buf0[12]+val32);
  data0_360448[(alu36+4225)] = (buf0[13]+val33);
  data0_360448[(alu36+4226)] = (buf0[14]+val34);
  data0_360448[(alu36+4227)] = (buf0[15]+val35);
}`;

const r_8_4_8_16_4_4_176_4_dispatch_140 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_1536:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_360448:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_180224:array<f32>;
@group(0) @binding(6)var<storage,read_write>data5_256:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 176; Lidx0++) {
    var cast0 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = ((gidx1*45056)+(lidx0*5632)+cast0);
    var val0 = data3_360448[(alu16+1)];
    var val1 = data3_360448[(alu16+1408)];
    var val2 = data3_360448[(alu16+1409)];
    var val3 = data3_360448[(alu16+1410)];
    var val4 = data3_360448[(alu16+1411)];
    var val5 = data3_360448[(alu16+2)];
    var val6 = data3_360448[(alu16+2112)];
    var val7 = data3_360448[(alu16+2113)];
    var val8 = data3_360448[(alu16+2114)];
    var val9 = data3_360448[(alu16+2115)];
    var val10 = data3_360448[(alu16+2816)];
    var val11 = data3_360448[(alu16+2817)];
    var val12 = data3_360448[(alu16+2818)];
    var val13 = data3_360448[(alu16+2819)];
    var val14 = data3_360448[(alu16+3)];
    var val15 = data3_360448[(alu16+3520)];
    var val16 = data3_360448[(alu16+3521)];
    var val17 = data3_360448[(alu16+3522)];
    var val18 = data3_360448[(alu16+3523)];
    var val19 = data3_360448[(alu16+4224)];
    var val20 = data3_360448[(alu16+4225)];
    var val21 = data3_360448[(alu16+4226)];
    var val22 = data3_360448[(alu16+4227)];
    var val23 = data3_360448[(alu16+4928)];
    var val24 = data3_360448[(alu16+4929)];
    var val25 = data3_360448[(alu16+4930)];
    var val26 = data3_360448[(alu16+4931)];
    var val27 = data3_360448[(alu16+704)];
    var val28 = data3_360448[(alu16+705)];
    var val29 = data3_360448[(alu16+706)];
    var val30 = data3_360448[(alu16+707)];
    var val31 = data3_360448[alu16];
    var alu17 = ((gidx0*45056)+(lidx1*2816)+cast0);
    var val32 = data4_180224[(alu17+1)];
    var val33 = data4_180224[(alu17+1408)];
    var val34 = data4_180224[(alu17+1409)];
    var val35 = data4_180224[(alu17+1410)];
    var val36 = data4_180224[(alu17+1411)];
    var val37 = data4_180224[(alu17+2)];
    var val38 = data4_180224[(alu17+2112)];
    var val39 = data4_180224[(alu17+2113)];
    var val40 = data4_180224[(alu17+2114)];
    var val41 = data4_180224[(alu17+2115)];
    var val42 = data4_180224[(alu17+3)];
    var val43 = data4_180224[(alu17+704)];
    var val44 = data4_180224[(alu17+705)];
    var val45 = data4_180224[(alu17+706)];
    var val46 = data4_180224[(alu17+707)];
    var val47 = data4_180224[alu17];
    var alu18 = (val0*(1/(1.0f+exp2((val0*-1.4426950216293335f))))*val28);
    var alu19 = (val31*(1/(1.0f+exp2((val31*-1.4426950216293335f))))*val27);
    var alu20 = (val5*(1/(1.0f+exp2((val5*-1.4426950216293335f))))*val29);
    var alu21 = (val14*(1/(1.0f+exp2((val14*-1.4426950216293335f))))*val30);
    var alu22 = (val10*(1/(1.0f+exp2((val10*-1.4426950216293335f))))*val15);
    var alu23 = (val11*(1/(1.0f+exp2((val11*-1.4426950216293335f))))*val16);
    var alu24 = (val12*(1/(1.0f+exp2((val12*-1.4426950216293335f))))*val17);
    var alu25 = (val13*(1/(1.0f+exp2((val13*-1.4426950216293335f))))*val18);
    var alu26 = (val19*(1/(1.0f+exp2((val19*-1.4426950216293335f))))*val23);
    var alu27 = (val20*(1/(1.0f+exp2((val20*-1.4426950216293335f))))*val24);
    var alu28 = (val21*(1/(1.0f+exp2((val21*-1.4426950216293335f))))*val25);
    var alu29 = (val22*(1/(1.0f+exp2((val22*-1.4426950216293335f))))*val26);
    var alu30 = (val1*(1/(1.0f+exp2((val1*-1.4426950216293335f))))*val6);
    var alu31 = (val2*(1/(1.0f+exp2((val2*-1.4426950216293335f))))*val7);
    var alu32 = (val3*(1/(1.0f+exp2((val3*-1.4426950216293335f))))*val8);
    var alu33 = (val4*(1/(1.0f+exp2((val4*-1.4426950216293335f))))*val9);
    buf0[0] = (buf0[0]+(alu19*val47)+(alu18*val32)+(alu20*val37)+(alu21*val42));
    buf0[1] = (buf0[1]+(alu19*val43)+(alu18*val44)+(alu20*val45)+(alu21*val46));
    buf0[10] = (buf0[10]+(alu22*val33)+(alu23*val34)+(alu24*val35)+(alu25*val36));
    buf0[11] = (buf0[11]+(alu22*val38)+(alu23*val39)+(alu24*val40)+(alu25*val41));
    buf0[12] = (buf0[12]+(alu26*val47)+(alu27*val32)+(alu28*val37)+(alu29*val42));
    buf0[13] = (buf0[13]+(alu26*val43)+(alu27*val44)+(alu28*val45)+(alu29*val46));
    buf0[14] = (buf0[14]+(alu26*val33)+(alu27*val34)+(alu28*val35)+(alu29*val36));
    buf0[15] = (buf0[15]+(alu26*val38)+(alu27*val39)+(alu28*val40)+(alu29*val41));
    buf0[2] = (buf0[2]+(alu19*val33)+(alu18*val34)+(alu20*val35)+(alu21*val36));
    buf0[3] = (buf0[3]+(alu19*val38)+(alu18*val39)+(alu20*val40)+(alu21*val41));
    buf0[4] = (buf0[4]+(alu30*val47)+(alu31*val32)+(alu32*val37)+(alu33*val42));
    buf0[5] = (buf0[5]+(alu30*val43)+(alu31*val44)+(alu32*val45)+(alu33*val46));
    buf0[6] = (buf0[6]+(alu30*val33)+(alu31*val34)+(alu32*val35)+(alu33*val36));
    buf0[7] = (buf0[7]+(alu30*val38)+(alu31*val39)+(alu32*val40)+(alu33*val41));
    buf0[8] = (buf0[8]+(alu22*val47)+(alu23*val32)+(alu24*val37)+(alu25*val42));
    buf0[9] = (buf0[9]+(alu22*val43)+(alu23*val44)+(alu24*val45)+(alu25*val46));
  }
  var alu51 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu52 = (alu51+bitcast<i32>((bitcast<u32>(gidx1)<<13u))+bitcast<i32>((bitcast<u32>(lidx0)<<10u)));
  var val48 = data1_65536[alu52];
  var alu53 = (alu52+1);
  var val49 = data1_65536[alu53];
  var alu54 = (alu52+2);
  var val50 = data1_65536[alu54];
  var alu55 = (alu52+256);
  var val51 = data1_65536[alu55];
  var alu56 = (alu52+257);
  var val52 = data1_65536[alu56];
  var alu57 = (alu52+258);
  var val53 = data1_65536[alu57];
  var alu58 = (alu52+259);
  var val54 = data1_65536[alu58];
  var alu59 = (alu52+3);
  var val55 = data1_65536[alu59];
  var alu60 = (alu52+512);
  var val56 = data1_65536[alu60];
  var alu61 = (alu52+513);
  var val57 = data1_65536[alu61];
  var alu62 = (alu52+514);
  var val58 = data1_65536[alu62];
  var alu63 = (alu52+515);
  var val59 = data1_65536[alu63];
  var alu64 = (alu52+768);
  var val60 = data1_65536[alu64];
  var alu65 = (alu52+769);
  var val61 = data1_65536[alu65];
  var alu66 = (alu52+770);
  var val62 = data1_65536[alu66];
  var alu67 = (alu52+771);
  var val63 = data1_65536[alu67];
  var val64 = data2_1536[(alu51+1280)];
  var val65 = data2_1536[(alu51+1281)];
  var val66 = data2_1536[(alu51+1282)];
  var val67 = data2_1536[(alu51+1283)];
  var val68 = data5_256[alu51];
  var val69 = data5_256[(alu51+1)];
  var val70 = data5_256[(alu51+2)];
  var val71 = data5_256[(alu51+3)];
  data0_65536[alu52] = (val48+(val64*(buf0[0]+val68)));
  data0_65536[alu53] = (val49+(val65*(buf0[1]+val69)));
  data0_65536[alu54] = (val50+(val66*(buf0[2]+val70)));
  data0_65536[alu55] = (val51+(val64*(buf0[4]+val68)));
  data0_65536[alu56] = (val52+(val65*(buf0[5]+val69)));
  data0_65536[alu57] = (val53+(val66*(buf0[6]+val70)));
  data0_65536[alu58] = (val54+(val67*(buf0[7]+val71)));
  data0_65536[alu59] = (val55+(val67*(buf0[3]+val71)));
  data0_65536[alu60] = (val56+(val64*(buf0[8]+val68)));
  data0_65536[alu61] = (val57+(val65*(buf0[9]+val69)));
  data0_65536[alu62] = (val58+(val66*(buf0[10]+val70)));
  data0_65536[alu63] = (val59+(val67*(buf0[11]+val71)));
  data0_65536[alu64] = (val60+(val64*(buf0[12]+val68)));
  data0_65536[alu65] = (val61+(val65*(buf0[13]+val69)));
  data0_65536[alu66] = (val62+(val66*(buf0[14]+val70)));
  data0_65536[alu67] = (val63+(val67*(buf0[15]+val71)));
}`;

const r_16_8_4_4_32_dispatch_141 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
var<workgroup> buf2: array<f32,128>;
@group(0) @binding(1)var<storage,read_write>data0_256:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@compute @workgroup_size(8,4) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,4>;
  var buf1: array<f32,4>;
  var gidx0 = i32(gindex.x); /* 16 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 32; Lidx0++) {
    var alu4 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast1<<8u))+bitcast<i32>((bitcast<u32>(Lidx0)<<3u))+lidx0);
    var val0 = data1_65536[alu4];
    var val1 = data1_65536[(alu4+1024)];
    var val2 = data1_65536[(alu4+2048)];
    var val3 = data1_65536[(alu4+3072)];
    buf0[0] = (buf0[0]+(val0*val0));
    buf0[1] = (buf0[1]+(val1*val1));
    buf0[2] = (buf0[2]+(val2*val2));
    buf0[3] = (buf0[3]+(val3*val3));
  }
  var cast2 = bitcast<i32>((cast1<<5u));
  var alu10 = (bitcast<i32>((bitcast<u32>(lidx0)<<2u))+cast2);
  buf2[alu10] = buf0[0];
  buf2[(alu10+1)] = buf0[1];
  buf2[(alu10+2)] = buf0[2];
  buf2[(alu10+3)] = buf0[3];
  workgroupBarrier();
  buf1[0] = 0.0f;
  buf1[1] = 0.0f;
  buf1[2] = 0.0f;
  buf1[3] = 0.0f;
  for (var Lidx102 = 0; Lidx102 < 8; Lidx102++) {
    var alu20 = (cast2+bitcast<i32>((bitcast<u32>(Lidx102)<<2u)));
    var val4 = buf2[alu20];
    var val5 = buf2[(alu20+1)];
    var val6 = buf2[(alu20+2)];
    var val7 = buf2[(alu20+3)];
    buf1[0] = (buf1[0]+val4);
    buf1[1] = (buf1[1]+val5);
    buf1[2] = (buf1[2]+val6);
    buf1[3] = (buf1[3]+val7);
  }
  var alu26 = (bitcast<i32>((cast0<<4u))+lidx1);
  var alu27 = (lidx0==0);
  if (alu27) {
    data0_256[alu26] = sqrt(((buf1[0]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+12)] = sqrt(((buf1[3]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+4)] = sqrt(((buf1[1]*0.00390625f)+9.999999974752427e-07f));
  }
  if (alu27) {
    data0_256[(alu26+8)] = sqrt(((buf1[2]*0.00390625f)+9.999999974752427e-07f));
  }
}`;

const E_8_4_8_16_4_4_dispatch_142 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_65536:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_256:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_256:array<f32>;
@group(0) @binding(5)var<storage,read_write>data4_512:array<f32>;
@compute @workgroup_size(8,16) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var gidx0 = i32(gindex.x); /* 4 */
  var gidx1 = i32(gindex.y); /* 8 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 16 */
  var cast0 = bitcast<u32>(gidx1);
  var cast1 = bitcast<u32>(lidx0);
  var alu0 = (bitcast<i32>((bitcast<u32>(gidx0)<<6u))+bitcast<i32>((bitcast<u32>(lidx1)<<2u)));
  var alu1 = (alu0+bitcast<i32>((cast0<<13u))+bitcast<i32>((cast1<<10u)));
  var val0 = data1_65536[alu1];
  var alu2 = (alu1+1);
  var val1 = data1_65536[alu2];
  var alu3 = (alu1+2);
  var val2 = data1_65536[alu3];
  var alu4 = (alu1+256);
  var val3 = data1_65536[alu4];
  var alu5 = (alu1+257);
  var val4 = data1_65536[alu5];
  var alu6 = (alu1+258);
  var val5 = data1_65536[alu6];
  var alu7 = (alu1+259);
  var val6 = data1_65536[alu7];
  var alu8 = (alu1+3);
  var val7 = data1_65536[alu8];
  var alu9 = (alu1+512);
  var val8 = data1_65536[alu9];
  var alu10 = (alu1+513);
  var val9 = data1_65536[alu10];
  var alu11 = (alu1+514);
  var val10 = data1_65536[alu11];
  var alu12 = (alu1+515);
  var val11 = data1_65536[alu12];
  var alu13 = (alu1+768);
  var val12 = data1_65536[alu13];
  var alu14 = (alu1+769);
  var val13 = data1_65536[alu14];
  var alu15 = (alu1+770);
  var val14 = data1_65536[alu15];
  var alu16 = (alu1+771);
  var val15 = data1_65536[alu16];
  var alu17 = (bitcast<i32>((cast0<<5u))+bitcast<i32>((cast1<<2u)));
  var val16 = data2_256[alu17];
  var val17 = data2_256[(alu17+1)];
  var val18 = data2_256[(alu17+2)];
  var val19 = data2_256[(alu17+3)];
  var val20 = data3_256[alu0];
  var alu18 = (alu0+1);
  var val21 = data3_256[alu18];
  var alu19 = (alu0+2);
  var val22 = data3_256[alu19];
  var alu20 = (alu0+3);
  var val23 = data3_256[alu20];
  var val24 = data4_512[alu0];
  var val25 = data4_512[alu18];
  var val26 = data4_512[alu19];
  var val27 = data4_512[(alu0+256)];
  var val28 = data4_512[(alu0+257)];
  var val29 = data4_512[(alu0+258)];
  var val30 = data4_512[(alu0+259)];
  var val31 = data4_512[alu20];
  var alu21 = (1/val16);
  var alu22 = (1/val17);
  var alu23 = (1/val18);
  var alu24 = (1/val19);
  var alu25 = (1.0f+val27);
  var alu26 = (1.0f+val28);
  var alu27 = (1.0f+val29);
  var alu28 = (1.0f+val30);
  data0_65536[alu1] = ((val0*alu21*val20*alu25)+val24);
  data0_65536[alu2] = ((val1*alu21*val21*alu26)+val25);
  data0_65536[alu3] = ((val2*alu21*val22*alu27)+val26);
  data0_65536[alu4] = ((val3*alu22*val20*alu25)+val24);
  data0_65536[alu5] = ((val4*alu22*val21*alu26)+val25);
  data0_65536[alu6] = ((val5*alu22*val22*alu27)+val26);
  data0_65536[alu7] = ((val6*alu22*val23*alu28)+val31);
  data0_65536[alu8] = ((val7*alu21*val23*alu28)+val31);
  data0_65536[alu9] = ((val8*alu23*val20*alu25)+val24);
  data0_65536[alu10] = ((val9*alu23*val21*alu26)+val25);
  data0_65536[alu11] = ((val10*alu23*val22*alu27)+val26);
  data0_65536[alu12] = ((val11*alu23*val23*alu28)+val31);
  data0_65536[alu13] = ((val12*alu24*val20*alu25)+val24);
  data0_65536[alu14] = ((val13*alu24*val21*alu26)+val25);
  data0_65536[alu15] = ((val14*alu24*val22*alu27)+val26);
  data0_65536[alu16] = ((val15*alu24*val23*alu28)+val31);
}`;

const r_3_16_8_4_2_4_4_64_4_dispatch_143 = `fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }
@group(0) @binding(0)
var<uniform> INFINITY : f32;
@group(0) @binding(1)var<storage,read_write>data0_49152:array<f32>;
@group(0) @binding(2)var<storage,read_write>data1_65536:array<f32>;
@group(0) @binding(3)var<storage,read_write>data2_49152:array<f32>;
@group(0) @binding(4)var<storage,read_write>data3_192:array<f32>;
@compute @workgroup_size(8,4,2) fn main(@builtin(workgroup_id) gindex: vec3<u32>,@builtin(local_invocation_id) lindex: vec3<u32>) {
  var buf0: array<f32,16>;
  var gidx0 = i32(gindex.x); /* 16 */
  var gidx1 = i32(gindex.y); /* 3 */
  var lidx0 = i32(lindex.x); /* 8 */
  var lidx1 = i32(lindex.y); /* 4 */
  var lidx2 = i32(lindex.z); /* 2 */
  var cast0 = bitcast<u32>(gidx0);
  var cast1 = bitcast<u32>(gidx1);
  var cast2 = bitcast<u32>(lidx1);
  buf0[0] = 0.0f;
  buf0[1] = 0.0f;
  buf0[10] = 0.0f;
  buf0[11] = 0.0f;
  buf0[12] = 0.0f;
  buf0[13] = 0.0f;
  buf0[14] = 0.0f;
  buf0[15] = 0.0f;
  buf0[2] = 0.0f;
  buf0[3] = 0.0f;
  buf0[4] = 0.0f;
  buf0[5] = 0.0f;
  buf0[6] = 0.0f;
  buf0[7] = 0.0f;
  buf0[8] = 0.0f;
  buf0[9] = 0.0f;
  for (var Lidx0 = 0; Lidx0 < 64; Lidx0++) {
    var cast3 = bitcast<i32>((bitcast<u32>(Lidx0)<<2u));
    var alu16 = (bitcast<i32>((cast0<<12u))+bitcast<i32>((cast2<<10u))+cast3);
    var val0 = data1_65536[alu16];
    var val1 = data1_65536[(alu16+1)];
    var val2 = data1_65536[(alu16+2)];
    var val3 = data1_65536[(alu16+256)];
    var val4 = data1_65536[(alu16+257)];
    var val5 = data1_65536[(alu16+258)];
    var val6 = data1_65536[(alu16+259)];
    var val7 = data1_65536[(alu16+3)];
    var val8 = data1_65536[(alu16+512)];
    var val9 = data1_65536[(alu16+513)];
    var val10 = data1_65536[(alu16+514)];
    var val11 = data1_65536[(alu16+515)];
    var val12 = data1_65536[(alu16+768)];
    var val13 = data1_65536[(alu16+769)];
    var val14 = data1_65536[(alu16+770)];
    var val15 = data1_65536[(alu16+771)];
    var alu17 = (bitcast<i32>((cast1<<8u))+(lidx2*3072)+(lidx0*6144)+cast3);
    var val16 = data2_49152[alu17];
    var val17 = data2_49152[(alu17+1)];
    var val18 = data2_49152[(alu17+1536)];
    var val19 = data2_49152[(alu17+1537)];
    var val20 = data2_49152[(alu17+1538)];
    var val21 = data2_49152[(alu17+1539)];
    var val22 = data2_49152[(alu17+2)];
    var val23 = data2_49152[(alu17+2304)];
    var val24 = data2_49152[(alu17+2305)];
    var val25 = data2_49152[(alu17+2306)];
    var val26 = data2_49152[(alu17+2307)];
    var val27 = data2_49152[(alu17+3)];
    var val28 = data2_49152[(alu17+768)];
    var val29 = data2_49152[(alu17+769)];
    var val30 = data2_49152[(alu17+770)];
    var val31 = data2_49152[(alu17+771)];
    buf0[0] = (buf0[0]+(val0*val16)+(val1*val17)+(val2*val22)+(val7*val27));
    buf0[1] = (buf0[1]+(val0*val28)+(val1*val29)+(val2*val30)+(val7*val31));
    buf0[10] = (buf0[10]+(val8*val18)+(val9*val19)+(val10*val20)+(val11*val21));
    buf0[11] = (buf0[11]+(val8*val23)+(val9*val24)+(val10*val25)+(val11*val26));
    buf0[12] = (buf0[12]+(val12*val16)+(val13*val17)+(val14*val22)+(val15*val27));
    buf0[13] = (buf0[13]+(val12*val28)+(val13*val29)+(val14*val30)+(val15*val31));
    buf0[14] = (buf0[14]+(val12*val18)+(val13*val19)+(val14*val20)+(val15*val21));
    buf0[15] = (buf0[15]+(val12*val23)+(val13*val24)+(val14*val25)+(val15*val26));
    buf0[2] = (buf0[2]+(val0*val18)+(val1*val19)+(val2*val20)+(val7*val21));
    buf0[3] = (buf0[3]+(val0*val23)+(val1*val24)+(val2*val25)+(val7*val26));
    buf0[4] = (buf0[4]+(val3*val16)+(val4*val17)+(val5*val22)+(val6*val27));
    buf0[5] = (buf0[5]+(val3*val28)+(val4*val29)+(val5*val30)+(val6*val31));
    buf0[6] = (buf0[6]+(val3*val18)+(val4*val19)+(val5*val20)+(val6*val21));
    buf0[7] = (buf0[7]+(val3*val23)+(val4*val24)+(val5*val25)+(val6*val26));
    buf0[8] = (buf0[8]+(val8*val16)+(val9*val17)+(val10*val22)+(val11*val27));
    buf0[9] = (buf0[9]+(val8*val28)+(val9*val29)+(val10*val30)+(val11*val31));
  }
  var alu35 = ((lidx2*12)+gidx1+(lidx0*24));
  var val32 = data3_192[(alu35+3)];
  var val33 = data3_192[(alu35+6)];
  var val34 = data3_192[(alu35+9)];
  var val35 = data3_192[alu35];
  var alu36 = (bitcast<i32>((cast0<<10u))+bitcast<i32>((bitcast<u32>(lidx0)<<7u))+bitcast<i32>((cast2<<5u))+bitcast<i32>((bitcast<u32>(lidx2)<<2u))+bitcast<i32>((cast1<<14u)));
  data0_49152[alu36] = (buf0[0]+val35);
  data0_49152[(alu36+1)] = (buf0[1]+val32);
  data0_49152[(alu36+10)] = (buf0[6]+val33);
  data0_49152[(alu36+11)] = (buf0[7]+val34);
  data0_49152[(alu36+16)] = (buf0[8]+val35);
  data0_49152[(alu36+17)] = (buf0[9]+val32);
  data0_49152[(alu36+18)] = (buf0[10]+val33);
  data0_49152[(alu36+19)] = (buf0[11]+val34);
  data0_49152[(alu36+2)] = (buf0[2]+val33);
  data0_49152[(alu36+24)] = (buf0[12]+val35);
  data0_49152[(alu36+25)] = (buf0[13]+val32);
  data0_49152[(alu36+26)] = (buf0[14]+val33);
  data0_49152[(alu36+27)] = (buf0[15]+val34);
  data0_49152[(alu36+3)] = (buf0[3]+val34);
  data0_49152[(alu36+8)] = (buf0[4]+val35);
  data0_49152[(alu36+9)] = (buf0[5]+val32);
}`;

const setupNet = async (device, safetensor) => {
    const metadata = getTensorMetadata(safetensor);
    const infinityBuf = createInfinityUniformBuf(device);

    const layouts=[device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]}),device.createBindGroupLayout({entries: [{binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' }}, {binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },{binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } }]})]

    const buf_0 = createEmptyBuf(device, 131072);;
    const input0 = createEmptyBuf(device, 196608);;
    const input3 = createEmptyBuf(device, 65536);;
    const input2 = createEmptyBuf(device, 196608);;
    const buf_1 = createWeightBuf(device, 229376, getTensorBuffer(safetensor, metadata['weights.patch_embed.0.weight']));
    const buf_2 = createEmptyBuf(device, 1024);;
    const input1 = createEmptyBuf(device, 4);;
    const buf_3 = createWeightBuf(device, 512, getTensorBuffer(safetensor, metadata['frequencies']));
    const buf_4 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['weights.time_embed.0.weight']));
    const buf_5 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.time_embed.0.bias']));
    const buf_6 = createEmptyBuf(device, 262144);;
    const buf_7 = createWeightBuf(device, 131072, getTensorBuffer(safetensor, metadata['weights.patch_embed.1.weight']));
    const buf_8 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.patch_embed.1.bias']));
    const buf_9 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['weights.positions']));
    const buf_10 = createEmptyBuf(device, 1024);;
    const buf_11 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['weights.time_embed.2.weight']));
    const buf_12 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.time_embed.2.bias']));
    const buf_13 = createEmptyBuf(device, 1024);;
    const buf_14 = createEmptyBuf(device, 6144);;
    const buf_15 = createWeightBuf(device, 1572864, getTensorBuffer(safetensor, metadata['weights.blocks.0.condition.1.weight']));
    const buf_16 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['weights.blocks.0.condition.1.bias']));
    const buf_17 = createEmptyBuf(device, 6144);;
    const buf_18 = createWeightBuf(device, 1572864, getTensorBuffer(safetensor, metadata['weights.blocks.1.condition.1.weight']));
    const buf_19 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['weights.blocks.1.condition.1.bias']));
    const buf_20 = createEmptyBuf(device, 6144);;
    const buf_21 = createWeightBuf(device, 1572864, getTensorBuffer(safetensor, metadata['weights.blocks.2.condition.1.weight']));
    const buf_22 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['weights.blocks.2.condition.1.bias']));
    const buf_23 = createEmptyBuf(device, 6144);;
    const buf_24 = createWeightBuf(device, 1572864, getTensorBuffer(safetensor, metadata['weights.blocks.3.condition.1.weight']));
    const buf_25 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['weights.blocks.3.condition.1.bias']));
    const buf_26 = createEmptyBuf(device, 6144);;
    const buf_27 = createWeightBuf(device, 1572864, getTensorBuffer(safetensor, metadata['weights.blocks.4.condition.1.weight']));
    const buf_28 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['weights.blocks.4.condition.1.bias']));
    const buf_29 = createEmptyBuf(device, 6144);;
    const buf_30 = createWeightBuf(device, 1572864, getTensorBuffer(safetensor, metadata['weights.blocks.5.condition.1.weight']));
    const buf_31 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['weights.blocks.5.condition.1.bias']));
    const buf_32 = createEmptyBuf(device, 6144);;
    const buf_33 = createWeightBuf(device, 1572864, getTensorBuffer(safetensor, metadata['weights.blocks.6.condition.1.weight']));
    const buf_34 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['weights.blocks.6.condition.1.bias']));
    const buf_35 = createEmptyBuf(device, 6144);;
    const buf_36 = createWeightBuf(device, 1572864, getTensorBuffer(safetensor, metadata['weights.blocks.7.condition.1.weight']));
    const buf_37 = createWeightBuf(device, 6144, getTensorBuffer(safetensor, metadata['weights.blocks.7.condition.1.bias']));
    const buf_38 = createEmptyBuf(device, 2048);;
    const buf_39 = createWeightBuf(device, 524288, getTensorBuffer(safetensor, metadata['weights.final_condition.1.weight']));
    const buf_40 = createWeightBuf(device, 2048, getTensorBuffer(safetensor, metadata['weights.final_condition.1.bias']));
    const buf_41 = createEmptyBuf(device, 262144);;
    const buf_42 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.0.norm_attention.weight']));
    const buf_43 = createEmptyBuf(device, 786432);;
    const buf_44 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['weights.blocks.0.attention.qkv.weight']));
    const buf_45 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['weights.blocks.0.attention.qkv.bias']));
    const buf_46 = createEmptyBuf(device, 8192);;
    const buf_47 = createEmptyBuf(device, 8192);;
    const buf_48 = createEmptyBuf(device, 262144);;
    const buf_49 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.0.attention.q_norm.weight']));
    const buf_50 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.0.attention.cosines']));
    const buf_51 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.0.attention.sines']));
    const buf_52 = createEmptyBuf(device, 262144);;
    const buf_53 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.0.attention.k_norm.weight']));
    const buf_54 = createEmptyBuf(device, 2097152);;
    const buf_55 = createEmptyBuf(device, 8192);;
    const buf_56 = createEmptyBuf(device, 8192);;
    const buf_57 = createEmptyBuf(device, 262144);;
    const buf_58 = createEmptyBuf(device, 262144);;
    const buf_59 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['weights.blocks.0.attention.output.weight']));
    const buf_60 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.0.attention.output.bias']));
    const buf_61 = createEmptyBuf(device, 1024);;
    const buf_62 = createEmptyBuf(device, 262144);;
    const buf_63 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.0.norm_mlp.weight']));
    const buf_64 = createEmptyBuf(device, 1441792);;
    const buf_65 = createWeightBuf(device, 1441792, getTensorBuffer(safetensor, metadata['weights.blocks.0.mlp.input.weight']));
    const buf_66 = createWeightBuf(device, 5632, getTensorBuffer(safetensor, metadata['weights.blocks.0.mlp.input.bias']));
    const buf_67 = createEmptyBuf(device, 262144);;
    const buf_68 = createWeightBuf(device, 720896, getTensorBuffer(safetensor, metadata['weights.blocks.0.mlp.output.weight']));
    const buf_69 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.0.mlp.output.bias']));
    const buf_70 = createEmptyBuf(device, 1024);;
    const buf_71 = createEmptyBuf(device, 262144);;
    const buf_72 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.1.norm_attention.weight']));
    const buf_73 = createEmptyBuf(device, 786432);;
    const buf_74 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['weights.blocks.1.attention.qkv.weight']));
    const buf_75 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['weights.blocks.1.attention.qkv.bias']));
    const buf_76 = createEmptyBuf(device, 8192);;
    const buf_77 = createEmptyBuf(device, 8192);;
    const buf_78 = createEmptyBuf(device, 262144);;
    const buf_79 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.1.attention.q_norm.weight']));
    const buf_80 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.1.attention.cosines']));
    const buf_81 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.1.attention.sines']));
    const buf_82 = createEmptyBuf(device, 262144);;
    const buf_83 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.1.attention.k_norm.weight']));
    const buf_84 = createEmptyBuf(device, 2097152);;
    const buf_85 = createEmptyBuf(device, 8192);;
    const buf_86 = createEmptyBuf(device, 8192);;
    const buf_87 = createEmptyBuf(device, 262144);;
    const buf_88 = createEmptyBuf(device, 262144);;
    const buf_89 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['weights.blocks.1.attention.output.weight']));
    const buf_90 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.1.attention.output.bias']));
    const buf_91 = createEmptyBuf(device, 1024);;
    const buf_92 = createEmptyBuf(device, 262144);;
    const buf_93 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.1.norm_mlp.weight']));
    const buf_94 = createEmptyBuf(device, 1441792);;
    const buf_95 = createWeightBuf(device, 1441792, getTensorBuffer(safetensor, metadata['weights.blocks.1.mlp.input.weight']));
    const buf_96 = createWeightBuf(device, 5632, getTensorBuffer(safetensor, metadata['weights.blocks.1.mlp.input.bias']));
    const buf_97 = createEmptyBuf(device, 262144);;
    const buf_98 = createWeightBuf(device, 720896, getTensorBuffer(safetensor, metadata['weights.blocks.1.mlp.output.weight']));
    const buf_99 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.1.mlp.output.bias']));
    const buf_100 = createEmptyBuf(device, 1024);;
    const buf_101 = createEmptyBuf(device, 262144);;
    const buf_102 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.2.norm_attention.weight']));
    const buf_103 = createEmptyBuf(device, 786432);;
    const buf_104 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['weights.blocks.2.attention.qkv.weight']));
    const buf_105 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['weights.blocks.2.attention.qkv.bias']));
    const buf_106 = createEmptyBuf(device, 8192);;
    const buf_107 = createEmptyBuf(device, 8192);;
    const buf_108 = createEmptyBuf(device, 262144);;
    const buf_109 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.2.attention.q_norm.weight']));
    const buf_110 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.2.attention.cosines']));
    const buf_111 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.2.attention.sines']));
    const buf_112 = createEmptyBuf(device, 262144);;
    const buf_113 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.2.attention.k_norm.weight']));
    const buf_114 = createEmptyBuf(device, 2097152);;
    const buf_115 = createEmptyBuf(device, 8192);;
    const buf_116 = createEmptyBuf(device, 8192);;
    const buf_117 = createEmptyBuf(device, 262144);;
    const buf_118 = createEmptyBuf(device, 262144);;
    const buf_119 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['weights.blocks.2.attention.output.weight']));
    const buf_120 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.2.attention.output.bias']));
    const buf_121 = createEmptyBuf(device, 1024);;
    const buf_122 = createEmptyBuf(device, 262144);;
    const buf_123 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.2.norm_mlp.weight']));
    const buf_124 = createEmptyBuf(device, 1441792);;
    const buf_125 = createWeightBuf(device, 1441792, getTensorBuffer(safetensor, metadata['weights.blocks.2.mlp.input.weight']));
    const buf_126 = createWeightBuf(device, 5632, getTensorBuffer(safetensor, metadata['weights.blocks.2.mlp.input.bias']));
    const buf_127 = createEmptyBuf(device, 262144);;
    const buf_128 = createWeightBuf(device, 720896, getTensorBuffer(safetensor, metadata['weights.blocks.2.mlp.output.weight']));
    const buf_129 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.2.mlp.output.bias']));
    const buf_130 = createEmptyBuf(device, 1024);;
    const buf_131 = createEmptyBuf(device, 262144);;
    const buf_132 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.3.norm_attention.weight']));
    const buf_133 = createEmptyBuf(device, 786432);;
    const buf_134 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['weights.blocks.3.attention.qkv.weight']));
    const buf_135 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['weights.blocks.3.attention.qkv.bias']));
    const buf_136 = createEmptyBuf(device, 8192);;
    const buf_137 = createEmptyBuf(device, 8192);;
    const buf_138 = createEmptyBuf(device, 262144);;
    const buf_139 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.3.attention.q_norm.weight']));
    const buf_140 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.3.attention.cosines']));
    const buf_141 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.3.attention.sines']));
    const buf_142 = createEmptyBuf(device, 262144);;
    const buf_143 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.3.attention.k_norm.weight']));
    const buf_144 = createEmptyBuf(device, 2097152);;
    const buf_145 = createEmptyBuf(device, 8192);;
    const buf_146 = createEmptyBuf(device, 8192);;
    const buf_147 = createEmptyBuf(device, 262144);;
    const buf_148 = createEmptyBuf(device, 262144);;
    const buf_149 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['weights.blocks.3.attention.output.weight']));
    const buf_150 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.3.attention.output.bias']));
    const buf_151 = createEmptyBuf(device, 1024);;
    const buf_152 = createEmptyBuf(device, 262144);;
    const buf_153 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.3.norm_mlp.weight']));
    const buf_154 = createEmptyBuf(device, 1441792);;
    const buf_155 = createWeightBuf(device, 1441792, getTensorBuffer(safetensor, metadata['weights.blocks.3.mlp.input.weight']));
    const buf_156 = createWeightBuf(device, 5632, getTensorBuffer(safetensor, metadata['weights.blocks.3.mlp.input.bias']));
    const buf_157 = createEmptyBuf(device, 262144);;
    const buf_158 = createWeightBuf(device, 720896, getTensorBuffer(safetensor, metadata['weights.blocks.3.mlp.output.weight']));
    const buf_159 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.3.mlp.output.bias']));
    const buf_160 = createEmptyBuf(device, 1024);;
    const buf_161 = createEmptyBuf(device, 262144);;
    const buf_162 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.4.norm_attention.weight']));
    const buf_163 = createEmptyBuf(device, 786432);;
    const buf_164 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['weights.blocks.4.attention.qkv.weight']));
    const buf_165 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['weights.blocks.4.attention.qkv.bias']));
    const buf_166 = createEmptyBuf(device, 8192);;
    const buf_167 = createEmptyBuf(device, 8192);;
    const buf_168 = createEmptyBuf(device, 262144);;
    const buf_169 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.4.attention.q_norm.weight']));
    const buf_170 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.4.attention.cosines']));
    const buf_171 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.4.attention.sines']));
    const buf_172 = createEmptyBuf(device, 262144);;
    const buf_173 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.4.attention.k_norm.weight']));
    const buf_174 = createEmptyBuf(device, 2097152);;
    const buf_175 = createEmptyBuf(device, 8192);;
    const buf_176 = createEmptyBuf(device, 8192);;
    const buf_177 = createEmptyBuf(device, 262144);;
    const buf_178 = createEmptyBuf(device, 262144);;
    const buf_179 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['weights.blocks.4.attention.output.weight']));
    const buf_180 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.4.attention.output.bias']));
    const buf_181 = createEmptyBuf(device, 1024);;
    const buf_182 = createEmptyBuf(device, 262144);;
    const buf_183 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.4.norm_mlp.weight']));
    const buf_184 = createEmptyBuf(device, 1441792);;
    const buf_185 = createWeightBuf(device, 1441792, getTensorBuffer(safetensor, metadata['weights.blocks.4.mlp.input.weight']));
    const buf_186 = createWeightBuf(device, 5632, getTensorBuffer(safetensor, metadata['weights.blocks.4.mlp.input.bias']));
    const buf_187 = createEmptyBuf(device, 262144);;
    const buf_188 = createWeightBuf(device, 720896, getTensorBuffer(safetensor, metadata['weights.blocks.4.mlp.output.weight']));
    const buf_189 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.4.mlp.output.bias']));
    const buf_190 = createEmptyBuf(device, 1024);;
    const buf_191 = createEmptyBuf(device, 262144);;
    const buf_192 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.5.norm_attention.weight']));
    const buf_193 = createEmptyBuf(device, 786432);;
    const buf_194 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['weights.blocks.5.attention.qkv.weight']));
    const buf_195 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['weights.blocks.5.attention.qkv.bias']));
    const buf_196 = createEmptyBuf(device, 8192);;
    const buf_197 = createEmptyBuf(device, 8192);;
    const buf_198 = createEmptyBuf(device, 262144);;
    const buf_199 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.5.attention.q_norm.weight']));
    const buf_200 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.5.attention.cosines']));
    const buf_201 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.5.attention.sines']));
    const buf_202 = createEmptyBuf(device, 262144);;
    const buf_203 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.5.attention.k_norm.weight']));
    const buf_204 = createEmptyBuf(device, 2097152);;
    const buf_205 = createEmptyBuf(device, 8192);;
    const buf_206 = createEmptyBuf(device, 8192);;
    const buf_207 = createEmptyBuf(device, 262144);;
    const buf_208 = createEmptyBuf(device, 262144);;
    const buf_209 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['weights.blocks.5.attention.output.weight']));
    const buf_210 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.5.attention.output.bias']));
    const buf_211 = createEmptyBuf(device, 1024);;
    const buf_212 = createEmptyBuf(device, 262144);;
    const buf_213 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.5.norm_mlp.weight']));
    const buf_214 = createEmptyBuf(device, 1441792);;
    const buf_215 = createWeightBuf(device, 1441792, getTensorBuffer(safetensor, metadata['weights.blocks.5.mlp.input.weight']));
    const buf_216 = createWeightBuf(device, 5632, getTensorBuffer(safetensor, metadata['weights.blocks.5.mlp.input.bias']));
    const buf_217 = createEmptyBuf(device, 262144);;
    const buf_218 = createWeightBuf(device, 720896, getTensorBuffer(safetensor, metadata['weights.blocks.5.mlp.output.weight']));
    const buf_219 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.5.mlp.output.bias']));
    const buf_220 = createEmptyBuf(device, 1024);;
    const buf_221 = createEmptyBuf(device, 262144);;
    const buf_222 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.6.norm_attention.weight']));
    const buf_223 = createEmptyBuf(device, 786432);;
    const buf_224 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['weights.blocks.6.attention.qkv.weight']));
    const buf_225 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['weights.blocks.6.attention.qkv.bias']));
    const buf_226 = createEmptyBuf(device, 8192);;
    const buf_227 = createEmptyBuf(device, 8192);;
    const buf_228 = createEmptyBuf(device, 262144);;
    const buf_229 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.6.attention.q_norm.weight']));
    const buf_230 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.6.attention.cosines']));
    const buf_231 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.6.attention.sines']));
    const buf_232 = createEmptyBuf(device, 262144);;
    const buf_233 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.6.attention.k_norm.weight']));
    const buf_234 = createEmptyBuf(device, 2097152);;
    const buf_235 = createEmptyBuf(device, 8192);;
    const buf_236 = createEmptyBuf(device, 8192);;
    const buf_237 = createEmptyBuf(device, 262144);;
    const buf_238 = createEmptyBuf(device, 262144);;
    const buf_239 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['weights.blocks.6.attention.output.weight']));
    const buf_240 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.6.attention.output.bias']));
    const buf_241 = createEmptyBuf(device, 1024);;
    const buf_242 = createEmptyBuf(device, 262144);;
    const buf_243 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.6.norm_mlp.weight']));
    const buf_244 = createEmptyBuf(device, 1441792);;
    const buf_245 = createWeightBuf(device, 1441792, getTensorBuffer(safetensor, metadata['weights.blocks.6.mlp.input.weight']));
    const buf_246 = createWeightBuf(device, 5632, getTensorBuffer(safetensor, metadata['weights.blocks.6.mlp.input.bias']));
    const buf_247 = createEmptyBuf(device, 262144);;
    const buf_248 = createWeightBuf(device, 720896, getTensorBuffer(safetensor, metadata['weights.blocks.6.mlp.output.weight']));
    const buf_249 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.6.mlp.output.bias']));
    const buf_250 = createEmptyBuf(device, 1024);;
    const buf_251 = createEmptyBuf(device, 262144);;
    const buf_252 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.7.norm_attention.weight']));
    const buf_253 = createEmptyBuf(device, 786432);;
    const buf_254 = createWeightBuf(device, 786432, getTensorBuffer(safetensor, metadata['weights.blocks.7.attention.qkv.weight']));
    const buf_255 = createWeightBuf(device, 3072, getTensorBuffer(safetensor, metadata['weights.blocks.7.attention.qkv.bias']));
    const buf_256 = createEmptyBuf(device, 8192);;
    const buf_257 = createEmptyBuf(device, 8192);;
    const buf_258 = createEmptyBuf(device, 262144);;
    const buf_259 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.7.attention.q_norm.weight']));
    const buf_260 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.7.attention.cosines']));
    const buf_261 = createWeightBuf(device, 32768, getTensorBuffer(safetensor, metadata['weights.blocks.7.attention.sines']));
    const buf_262 = createEmptyBuf(device, 262144);;
    const buf_263 = createWeightBuf(device, 128, getTensorBuffer(safetensor, metadata['weights.blocks.7.attention.k_norm.weight']));
    const buf_264 = createEmptyBuf(device, 2097152);;
    const buf_265 = createEmptyBuf(device, 8192);;
    const buf_266 = createEmptyBuf(device, 8192);;
    const buf_267 = createEmptyBuf(device, 262144);;
    const buf_268 = createEmptyBuf(device, 262144);;
    const buf_269 = createWeightBuf(device, 262144, getTensorBuffer(safetensor, metadata['weights.blocks.7.attention.output.weight']));
    const buf_270 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.7.attention.output.bias']));
    const buf_271 = createEmptyBuf(device, 1024);;
    const buf_272 = createEmptyBuf(device, 262144);;
    const buf_273 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.7.norm_mlp.weight']));
    const buf_274 = createEmptyBuf(device, 1441792);;
    const buf_275 = createWeightBuf(device, 1441792, getTensorBuffer(safetensor, metadata['weights.blocks.7.mlp.input.weight']));
    const buf_276 = createWeightBuf(device, 5632, getTensorBuffer(safetensor, metadata['weights.blocks.7.mlp.input.bias']));
    const buf_277 = createEmptyBuf(device, 262144);;
    const buf_278 = createWeightBuf(device, 720896, getTensorBuffer(safetensor, metadata['weights.blocks.7.mlp.output.weight']));
    const buf_279 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.blocks.7.mlp.output.bias']));
    const buf_280 = createEmptyBuf(device, 1024);;
    const buf_281 = createEmptyBuf(device, 262144);;
    const buf_282 = createWeightBuf(device, 1024, getTensorBuffer(safetensor, metadata['weights.final_norm.weight']));
    const output0 = createEmptyBuf(device, 196608);;
    const buf_283 = createWeightBuf(device, 196608, getTensorBuffer(safetensor, metadata['weights.final_projection.weight']));
    const buf_284 = createWeightBuf(device, 768, getTensorBuffer(safetensor, metadata['weights.final_projection.bias']));

    const gpuWriteBuffer0 = device.createBuffer({size:input0.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer1 = device.createBuffer({size:input1.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer2 = device.createBuffer({size:input2.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });
    const gpuWriteBuffer3 = device.createBuffer({size:input3.size, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.MAP_WRITE });

    const gpuReadBuffer0 = device.createBuffer({size:output0.size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

    const kernels = [r_16_2_16_4_4_4_7_8_8_dispatch_0, r_256_16_2_8_dispatch_1, r_8_4_8_16_4_4_32_4_dispatch_2, r_16_8_4_4_32_dispatch_3, r_16_8_4_4_32_dispatch_4, r_96_8_4_4_32_dispatch_5, r_96_8_4_4_32_dispatch_6, r_96_8_4_4_32_dispatch_7, r_96_8_4_4_32_dispatch_8, r_96_8_4_4_32_dispatch_9, r_96_8_4_4_32_dispatch_10, r_96_8_4_4_32_dispatch_11, r_96_8_4_4_32_dispatch_12, r_32_8_4_4_32_dispatch_13, E_8_4_8_16_4_4_dispatch_14, r_8_16_8_16_3_4_64_4_dispatch_15, r_8_16_8_4_4_4_dispatch_16, r_8_16_8_4_4_4_dispatch_17, E_2_32_8_16_2_4_dispatch_18, E_2_32_8_16_2_4_dispatch_19, r_8_8_4_8_16_4_4_32_dispatch_20, r_2048_16_16_dispatch_21, r_2048_16_16_dispatch_22, r_8_4_16_8_4_4_64_4_dispatch_23, r_8_4_8_16_4_4_8_32_dispatch_24, r_16_8_4_4_32_dispatch_25, E_8_4_8_16_4_4_dispatch_26, r_8_22_8_16_4_4_64_4_dispatch_27, r_8_4_8_16_4_4_176_4_dispatch_28, r_16_8_4_4_32_dispatch_29, E_8_4_8_16_4_4_dispatch_30, r_8_16_8_16_3_4_64_4_dispatch_31, r_8_16_8_4_4_4_dispatch_32, r_8_16_8_4_4_4_dispatch_33, E_2_32_8_16_2_4_dispatch_34, E_2_32_8_16_2_4_dispatch_35, r_8_8_4_8_16_4_4_32_dispatch_36, r_2048_16_16_dispatch_37, r_2048_16_16_dispatch_38, r_8_4_16_8_4_4_64_4_dispatch_39, r_8_4_8_16_4_4_8_32_dispatch_40, r_16_8_4_4_32_dispatch_41, E_8_4_8_16_4_4_dispatch_42, r_8_22_8_16_4_4_64_4_dispatch_43, r_8_4_8_16_4_4_176_4_dispatch_44, r_16_8_4_4_32_dispatch_45, E_8_4_8_16_4_4_dispatch_46, r_8_16_8_16_3_4_64_4_dispatch_47, r_8_16_8_4_4_4_dispatch_48, r_8_16_8_4_4_4_dispatch_49, E_2_32_8_16_2_4_dispatch_50, E_2_32_8_16_2_4_dispatch_51, r_8_8_4_8_16_4_4_32_dispatch_52, r_2048_16_16_dispatch_53, r_2048_16_16_dispatch_54, r_8_4_16_8_4_4_64_4_dispatch_55, r_8_4_8_16_4_4_8_32_dispatch_56, r_16_8_4_4_32_dispatch_57, E_8_4_8_16_4_4_dispatch_58, r_8_22_8_16_4_4_64_4_dispatch_59, r_8_4_8_16_4_4_176_4_dispatch_60, r_16_8_4_4_32_dispatch_61, E_8_4_8_16_4_4_dispatch_62, r_8_16_8_16_3_4_64_4_dispatch_63, r_8_16_8_4_4_4_dispatch_64, r_8_16_8_4_4_4_dispatch_65, E_2_32_8_16_2_4_dispatch_66, E_2_32_8_16_2_4_dispatch_67, r_8_8_4_8_16_4_4_32_dispatch_68, r_2048_16_16_dispatch_69, r_2048_16_16_dispatch_70, r_8_4_16_8_4_4_64_4_dispatch_71, r_8_4_8_16_4_4_8_32_dispatch_72, r_16_8_4_4_32_dispatch_73, E_8_4_8_16_4_4_dispatch_74, r_8_22_8_16_4_4_64_4_dispatch_75, r_8_4_8_16_4_4_176_4_dispatch_76, r_16_8_4_4_32_dispatch_77, E_8_4_8_16_4_4_dispatch_78, r_8_16_8_16_3_4_64_4_dispatch_79, r_8_16_8_4_4_4_dispatch_80, r_8_16_8_4_4_4_dispatch_81, E_2_32_8_16_2_4_dispatch_82, E_2_32_8_16_2_4_dispatch_83, r_8_8_4_8_16_4_4_32_dispatch_84, r_2048_16_16_dispatch_85, r_2048_16_16_dispatch_86, r_8_4_16_8_4_4_64_4_dispatch_87, r_8_4_8_16_4_4_8_32_dispatch_88, r_16_8_4_4_32_dispatch_89, E_8_4_8_16_4_4_dispatch_90, r_8_22_8_16_4_4_64_4_dispatch_91, r_8_4_8_16_4_4_176_4_dispatch_92, r_16_8_4_4_32_dispatch_93, E_8_4_8_16_4_4_dispatch_94, r_8_16_8_16_3_4_64_4_dispatch_95, r_8_16_8_4_4_4_dispatch_96, r_8_16_8_4_4_4_dispatch_97, E_2_32_8_16_2_4_dispatch_98, E_2_32_8_16_2_4_dispatch_99, r_8_8_4_8_16_4_4_32_dispatch_100, r_2048_16_16_dispatch_101, r_2048_16_16_dispatch_102, r_8_4_16_8_4_4_64_4_dispatch_103, r_8_4_8_16_4_4_8_32_dispatch_104, r_16_8_4_4_32_dispatch_105, E_8_4_8_16_4_4_dispatch_106, r_8_22_8_16_4_4_64_4_dispatch_107, r_8_4_8_16_4_4_176_4_dispatch_108, r_16_8_4_4_32_dispatch_109, E_8_4_8_16_4_4_dispatch_110, r_8_16_8_16_3_4_64_4_dispatch_111, r_8_16_8_4_4_4_dispatch_112, r_8_16_8_4_4_4_dispatch_113, E_2_32_8_16_2_4_dispatch_114, E_2_32_8_16_2_4_dispatch_115, r_8_8_4_8_16_4_4_32_dispatch_116, r_2048_16_16_dispatch_117, r_2048_16_16_dispatch_118, r_8_4_16_8_4_4_64_4_dispatch_119, r_8_4_8_16_4_4_8_32_dispatch_120, r_16_8_4_4_32_dispatch_121, E_8_4_8_16_4_4_dispatch_122, r_8_22_8_16_4_4_64_4_dispatch_123, r_8_4_8_16_4_4_176_4_dispatch_124, r_16_8_4_4_32_dispatch_125, E_8_4_8_16_4_4_dispatch_126, r_8_16_8_16_3_4_64_4_dispatch_127, r_8_16_8_4_4_4_dispatch_128, r_8_16_8_4_4_4_dispatch_129, E_2_32_8_16_2_4_dispatch_130, E_2_32_8_16_2_4_dispatch_131, r_8_8_4_8_16_4_4_32_dispatch_132, r_2048_16_16_dispatch_133, r_2048_16_16_dispatch_134, r_8_4_16_8_4_4_64_4_dispatch_135, r_8_4_8_16_4_4_8_32_dispatch_136, r_16_8_4_4_32_dispatch_137, E_8_4_8_16_4_4_dispatch_138, r_8_22_8_16_4_4_64_4_dispatch_139, r_8_4_8_16_4_4_176_4_dispatch_140, r_16_8_4_4_32_dispatch_141, E_8_4_8_16_4_4_dispatch_142, r_3_16_8_4_2_4_4_64_4_dispatch_143];
    const pipelines = await Promise.all(kernels.map(async (name, i) => {
      return await device.createComputePipelineAsync({
          layout: device.createPipelineLayout({
              bindGroupLayouts: [layouts[i]],
          }),
          compute: {
              module: device.createShaderModule({
                  code: name,
              }),
              entryPoint: "main",
          },
      });
  }))

    return async (_input0,_input1,_input2,_input3) => {
        const commandEncoder = device.createCommandEncoder();
        await gpuWriteBuffer0.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer0.getMappedRange()).set(_input0);
        gpuWriteBuffer0.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer0, 0, input0, 0, gpuWriteBuffer0.size);
    await gpuWriteBuffer1.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer1.getMappedRange()).set(_input1);
        gpuWriteBuffer1.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer1, 0, input1, 0, gpuWriteBuffer1.size);
    await gpuWriteBuffer2.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer2.getMappedRange()).set(_input2);
        gpuWriteBuffer2.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer2, 0, input2, 0, gpuWriteBuffer2.size);
    await gpuWriteBuffer3.mapAsync(GPUMapMode.WRITE);
        new Float32Array(gpuWriteBuffer3.getMappedRange()).set(_input3);
        gpuWriteBuffer3.unmap();
        commandEncoder.copyBufferToBuffer(gpuWriteBuffer3, 0, input3, 0, gpuWriteBuffer3.size);
        addComputePass(device, commandEncoder, pipelines[0], layouts[0], infinityBuf, [buf_0, input0, input3, input2, buf_1], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[1], layouts[1], infinityBuf, [buf_2, input1, buf_3, buf_4, buf_5], [256, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[2], layouts[2], infinityBuf, [buf_6, buf_0, buf_7, buf_8, buf_9], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[3], layouts[3], infinityBuf, [buf_10, buf_2, buf_11, buf_12], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[4], layouts[4], infinityBuf, [buf_13, buf_6], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[5], layouts[5], infinityBuf, [buf_14, buf_10, buf_15, buf_16], [96, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[6], layouts[6], infinityBuf, [buf_17, buf_10, buf_18, buf_19], [96, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[7], layouts[7], infinityBuf, [buf_20, buf_10, buf_21, buf_22], [96, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[8], layouts[8], infinityBuf, [buf_23, buf_10, buf_24, buf_25], [96, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[9], layouts[9], infinityBuf, [buf_26, buf_10, buf_27, buf_28], [96, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[10], layouts[10], infinityBuf, [buf_29, buf_10, buf_30, buf_31], [96, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[11], layouts[11], infinityBuf, [buf_32, buf_10, buf_33, buf_34], [96, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[12], layouts[12], infinityBuf, [buf_35, buf_10, buf_36, buf_37], [96, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[13], layouts[13], infinityBuf, [buf_38, buf_10, buf_39, buf_40], [32, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[14], layouts[14], infinityBuf, [buf_41, buf_6, buf_13, buf_42, buf_14], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[15], layouts[15], infinityBuf, [buf_43, buf_41, buf_44, buf_45], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[16], layouts[16], infinityBuf, [buf_46, buf_43], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[17], layouts[17], infinityBuf, [buf_47, buf_43], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[18], layouts[18], infinityBuf, [buf_48, buf_43, buf_46, buf_49, buf_50, buf_51], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[19], layouts[19], infinityBuf, [buf_52, buf_43, buf_47, buf_53, buf_50, buf_51], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[20], layouts[20], infinityBuf, [buf_54, buf_48, buf_52], [4, 8, 8]);
        addComputePass(device, commandEncoder, pipelines[21], layouts[21], infinityBuf, [buf_55, buf_54], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[22], layouts[22], infinityBuf, [buf_56, buf_54, buf_55], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[23], layouts[23], infinityBuf, [buf_57, buf_54, buf_55, buf_56, buf_43], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[24], layouts[24], infinityBuf, [buf_58, buf_6, buf_14, buf_57, buf_59, buf_60], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[25], layouts[25], infinityBuf, [buf_61, buf_58], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[26], layouts[26], infinityBuf, [buf_62, buf_58, buf_61, buf_63, buf_14], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[27], layouts[27], infinityBuf, [buf_64, buf_62, buf_65, buf_66], [22, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[28], layouts[28], infinityBuf, [buf_67, buf_58, buf_14, buf_64, buf_68, buf_69], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[29], layouts[29], infinityBuf, [buf_70, buf_67], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[30], layouts[30], infinityBuf, [buf_71, buf_67, buf_70, buf_72, buf_17], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[31], layouts[31], infinityBuf, [buf_73, buf_71, buf_74, buf_75], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[32], layouts[32], infinityBuf, [buf_76, buf_73], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[33], layouts[33], infinityBuf, [buf_77, buf_73], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[34], layouts[34], infinityBuf, [buf_78, buf_73, buf_76, buf_79, buf_80, buf_81], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[35], layouts[35], infinityBuf, [buf_82, buf_73, buf_77, buf_83, buf_80, buf_81], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[36], layouts[36], infinityBuf, [buf_84, buf_78, buf_82], [4, 8, 8]);
        addComputePass(device, commandEncoder, pipelines[37], layouts[37], infinityBuf, [buf_85, buf_84], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[38], layouts[38], infinityBuf, [buf_86, buf_84, buf_85], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[39], layouts[39], infinityBuf, [buf_87, buf_84, buf_85, buf_86, buf_73], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[40], layouts[40], infinityBuf, [buf_88, buf_67, buf_17, buf_87, buf_89, buf_90], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[41], layouts[41], infinityBuf, [buf_91, buf_88], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[42], layouts[42], infinityBuf, [buf_92, buf_88, buf_91, buf_93, buf_17], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[43], layouts[43], infinityBuf, [buf_94, buf_92, buf_95, buf_96], [22, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[44], layouts[44], infinityBuf, [buf_97, buf_88, buf_17, buf_94, buf_98, buf_99], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[45], layouts[45], infinityBuf, [buf_100, buf_97], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[46], layouts[46], infinityBuf, [buf_101, buf_97, buf_100, buf_102, buf_20], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[47], layouts[47], infinityBuf, [buf_103, buf_101, buf_104, buf_105], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[48], layouts[48], infinityBuf, [buf_106, buf_103], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[49], layouts[49], infinityBuf, [buf_107, buf_103], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[50], layouts[50], infinityBuf, [buf_108, buf_103, buf_106, buf_109, buf_110, buf_111], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[51], layouts[51], infinityBuf, [buf_112, buf_103, buf_107, buf_113, buf_110, buf_111], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[52], layouts[52], infinityBuf, [buf_114, buf_108, buf_112], [4, 8, 8]);
        addComputePass(device, commandEncoder, pipelines[53], layouts[53], infinityBuf, [buf_115, buf_114], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[54], layouts[54], infinityBuf, [buf_116, buf_114, buf_115], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[55], layouts[55], infinityBuf, [buf_117, buf_114, buf_115, buf_116, buf_103], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[56], layouts[56], infinityBuf, [buf_118, buf_97, buf_20, buf_117, buf_119, buf_120], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[57], layouts[57], infinityBuf, [buf_121, buf_118], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[58], layouts[58], infinityBuf, [buf_122, buf_118, buf_121, buf_123, buf_20], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[59], layouts[59], infinityBuf, [buf_124, buf_122, buf_125, buf_126], [22, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[60], layouts[60], infinityBuf, [buf_127, buf_118, buf_20, buf_124, buf_128, buf_129], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[61], layouts[61], infinityBuf, [buf_130, buf_127], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[62], layouts[62], infinityBuf, [buf_131, buf_127, buf_130, buf_132, buf_23], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[63], layouts[63], infinityBuf, [buf_133, buf_131, buf_134, buf_135], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[64], layouts[64], infinityBuf, [buf_136, buf_133], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[65], layouts[65], infinityBuf, [buf_137, buf_133], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[66], layouts[66], infinityBuf, [buf_138, buf_133, buf_136, buf_139, buf_140, buf_141], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[67], layouts[67], infinityBuf, [buf_142, buf_133, buf_137, buf_143, buf_140, buf_141], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[68], layouts[68], infinityBuf, [buf_144, buf_138, buf_142], [4, 8, 8]);
        addComputePass(device, commandEncoder, pipelines[69], layouts[69], infinityBuf, [buf_145, buf_144], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[70], layouts[70], infinityBuf, [buf_146, buf_144, buf_145], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[71], layouts[71], infinityBuf, [buf_147, buf_144, buf_145, buf_146, buf_133], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[72], layouts[72], infinityBuf, [buf_148, buf_127, buf_23, buf_147, buf_149, buf_150], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[73], layouts[73], infinityBuf, [buf_151, buf_148], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[74], layouts[74], infinityBuf, [buf_152, buf_148, buf_151, buf_153, buf_23], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[75], layouts[75], infinityBuf, [buf_154, buf_152, buf_155, buf_156], [22, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[76], layouts[76], infinityBuf, [buf_157, buf_148, buf_23, buf_154, buf_158, buf_159], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[77], layouts[77], infinityBuf, [buf_160, buf_157], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[78], layouts[78], infinityBuf, [buf_161, buf_157, buf_160, buf_162, buf_26], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[79], layouts[79], infinityBuf, [buf_163, buf_161, buf_164, buf_165], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[80], layouts[80], infinityBuf, [buf_166, buf_163], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[81], layouts[81], infinityBuf, [buf_167, buf_163], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[82], layouts[82], infinityBuf, [buf_168, buf_163, buf_166, buf_169, buf_170, buf_171], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[83], layouts[83], infinityBuf, [buf_172, buf_163, buf_167, buf_173, buf_170, buf_171], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[84], layouts[84], infinityBuf, [buf_174, buf_168, buf_172], [4, 8, 8]);
        addComputePass(device, commandEncoder, pipelines[85], layouts[85], infinityBuf, [buf_175, buf_174], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[86], layouts[86], infinityBuf, [buf_176, buf_174, buf_175], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[87], layouts[87], infinityBuf, [buf_177, buf_174, buf_175, buf_176, buf_163], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[88], layouts[88], infinityBuf, [buf_178, buf_157, buf_26, buf_177, buf_179, buf_180], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[89], layouts[89], infinityBuf, [buf_181, buf_178], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[90], layouts[90], infinityBuf, [buf_182, buf_178, buf_181, buf_183, buf_26], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[91], layouts[91], infinityBuf, [buf_184, buf_182, buf_185, buf_186], [22, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[92], layouts[92], infinityBuf, [buf_187, buf_178, buf_26, buf_184, buf_188, buf_189], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[93], layouts[93], infinityBuf, [buf_190, buf_187], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[94], layouts[94], infinityBuf, [buf_191, buf_187, buf_190, buf_192, buf_29], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[95], layouts[95], infinityBuf, [buf_193, buf_191, buf_194, buf_195], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[96], layouts[96], infinityBuf, [buf_196, buf_193], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[97], layouts[97], infinityBuf, [buf_197, buf_193], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[98], layouts[98], infinityBuf, [buf_198, buf_193, buf_196, buf_199, buf_200, buf_201], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[99], layouts[99], infinityBuf, [buf_202, buf_193, buf_197, buf_203, buf_200, buf_201], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[100], layouts[100], infinityBuf, [buf_204, buf_198, buf_202], [4, 8, 8]);
        addComputePass(device, commandEncoder, pipelines[101], layouts[101], infinityBuf, [buf_205, buf_204], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[102], layouts[102], infinityBuf, [buf_206, buf_204, buf_205], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[103], layouts[103], infinityBuf, [buf_207, buf_204, buf_205, buf_206, buf_193], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[104], layouts[104], infinityBuf, [buf_208, buf_187, buf_29, buf_207, buf_209, buf_210], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[105], layouts[105], infinityBuf, [buf_211, buf_208], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[106], layouts[106], infinityBuf, [buf_212, buf_208, buf_211, buf_213, buf_29], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[107], layouts[107], infinityBuf, [buf_214, buf_212, buf_215, buf_216], [22, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[108], layouts[108], infinityBuf, [buf_217, buf_208, buf_29, buf_214, buf_218, buf_219], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[109], layouts[109], infinityBuf, [buf_220, buf_217], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[110], layouts[110], infinityBuf, [buf_221, buf_217, buf_220, buf_222, buf_32], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[111], layouts[111], infinityBuf, [buf_223, buf_221, buf_224, buf_225], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[112], layouts[112], infinityBuf, [buf_226, buf_223], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[113], layouts[113], infinityBuf, [buf_227, buf_223], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[114], layouts[114], infinityBuf, [buf_228, buf_223, buf_226, buf_229, buf_230, buf_231], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[115], layouts[115], infinityBuf, [buf_232, buf_223, buf_227, buf_233, buf_230, buf_231], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[116], layouts[116], infinityBuf, [buf_234, buf_228, buf_232], [4, 8, 8]);
        addComputePass(device, commandEncoder, pipelines[117], layouts[117], infinityBuf, [buf_235, buf_234], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[118], layouts[118], infinityBuf, [buf_236, buf_234, buf_235], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[119], layouts[119], infinityBuf, [buf_237, buf_234, buf_235, buf_236, buf_223], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[120], layouts[120], infinityBuf, [buf_238, buf_217, buf_32, buf_237, buf_239, buf_240], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[121], layouts[121], infinityBuf, [buf_241, buf_238], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[122], layouts[122], infinityBuf, [buf_242, buf_238, buf_241, buf_243, buf_32], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[123], layouts[123], infinityBuf, [buf_244, buf_242, buf_245, buf_246], [22, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[124], layouts[124], infinityBuf, [buf_247, buf_238, buf_32, buf_244, buf_248, buf_249], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[125], layouts[125], infinityBuf, [buf_250, buf_247], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[126], layouts[126], infinityBuf, [buf_251, buf_247, buf_250, buf_252, buf_35], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[127], layouts[127], infinityBuf, [buf_253, buf_251, buf_254, buf_255], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[128], layouts[128], infinityBuf, [buf_256, buf_253], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[129], layouts[129], infinityBuf, [buf_257, buf_253], [16, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[130], layouts[130], infinityBuf, [buf_258, buf_253, buf_256, buf_259, buf_260, buf_261], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[131], layouts[131], infinityBuf, [buf_262, buf_253, buf_257, buf_263, buf_260, buf_261], [32, 2, 1]);
        addComputePass(device, commandEncoder, pipelines[132], layouts[132], infinityBuf, [buf_264, buf_258, buf_262], [4, 8, 8]);
        addComputePass(device, commandEncoder, pipelines[133], layouts[133], infinityBuf, [buf_265, buf_264], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[134], layouts[134], infinityBuf, [buf_266, buf_264, buf_265], [2048, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[135], layouts[135], infinityBuf, [buf_267, buf_264, buf_265, buf_266, buf_253], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[136], layouts[136], infinityBuf, [buf_268, buf_247, buf_35, buf_267, buf_269, buf_270], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[137], layouts[137], infinityBuf, [buf_271, buf_268], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[138], layouts[138], infinityBuf, [buf_272, buf_268, buf_271, buf_273, buf_35], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[139], layouts[139], infinityBuf, [buf_274, buf_272, buf_275, buf_276], [22, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[140], layouts[140], infinityBuf, [buf_277, buf_268, buf_35, buf_274, buf_278, buf_279], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[141], layouts[141], infinityBuf, [buf_280, buf_277], [16, 1, 1]);
        addComputePass(device, commandEncoder, pipelines[142], layouts[142], infinityBuf, [buf_281, buf_277, buf_280, buf_282, buf_38], [4, 8, 1]);
        addComputePass(device, commandEncoder, pipelines[143], layouts[143], infinityBuf, [output0, buf_281, buf_283, buf_284], [16, 3, 1]);
        commandEncoder.copyBufferToBuffer(output0, 0, gpuReadBuffer0, 0, output0.size);
        const gpuCommands = commandEncoder.finish();
        device.queue.submit([gpuCommands]);

        await gpuReadBuffer0.mapAsync(GPUMapMode.READ);
        const resultBuffer0 = new Float32Array(gpuReadBuffer0.size/4);
        resultBuffer0.set(new Float32Array(gpuReadBuffer0.getMappedRange()));
        gpuReadBuffer0.unmap();
        return [resultBuffer0];
    }
}
const load = async (device, weight_path) => { return await fetch(weight_path).then(x => x.arrayBuffer()).then(x => setupNet(device, new Uint8Array(x))); }
return { load, setupNet };
})();
export default mandelbrot;
