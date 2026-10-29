'use strict';
// Known numerical comparisons, not screenshots of the implementation.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../tools/image_compare.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script); // Parse all browser code, including the UI.
const workerCode = script.slice(script.indexOf('function comparePixels('), script.indexOf('\nconst $='));
let result;
const context = vm.createContext({self:{postMessage(value){result=value;}}});
vm.runInContext(workerCode, context);
function pixels(w,h,color){return Uint8ClampedArray.from(Array(w*h).fill(color).flat());}
function compare(a,b,w,h,options={}){
 result=null;
 context.event={data:{width:w,height:h,a:a.buffer,b:b.buffer,margins:{top:0,right:0,bottom:0,left:0},threshold:0,gain:8,...options}};
 vm.runInContext('comparePixels(event)',context);
 return result;
}
function near(a,b,epsilon=1e-9){assert.ok(Math.abs(a-b)<epsilon,`${a} != ${b}`);}
let a=pixels(9,9,[50,100,150,255]);
let r=compare(a,a,9,9);
assert.equal(r.metrics.pixel_count,81);
assert.equal(r.metrics.rgb_mae_255,0);
assert.equal(r.metrics.rgb_rmse_255,0);
assert.equal(r.metrics.rgb_psnr_db,null);
assert.equal(r.metrics.rgb_identical,true);
near(r.metrics.ssim_luma_8x8,1);
assert.equal(r.metrics.rgb_changed_pixels,0);
assert.equal(r.metrics.alpha_mae_255,0);
let b=pixels(9,9,[60,110,160,255]);
r=compare(a,b,9,9,{threshold:10,gain:4});
near(r.metrics.rgb_mae_255,10);
near(r.metrics.rgb_rmse_255,10);
near(r.metrics.rgb_psnr_db,20*Math.log10(255/10));
assert.equal(r.metrics.rgb_changed_pixels,0); // Strictly greater than threshold.
assert.equal(r.metrics.rgb_max_channel_error_255,10);
assert.deepEqual(Array.from(new Uint8ClampedArray(r.diff).slice(0,4)),[40,40,40,255]);
r=compare(a,b,9,9,{threshold:9});
assert.equal(r.metrics.rgb_changed_pixels,81);
// The final partial block must not disappear from SSIM or RGB scoring.
a=pixels(9,9,[0,0,0,255]);b=a.slice();b.set([255,255,255,255],80*4);
r=compare(a,b,9,9);
assert.equal(r.metrics.rgb_changed_pixels,1);
near(r.metrics.rgb_mae_255,255/81);
near(r.metrics.ssim_luma_8x8,(80+2.55**2/(255**2+2.55**2))/81);
// Excluded borders do not contribute, and remain black in the saved diff.
r=compare(a,b,9,9,{margins:{top:0,right:1,bottom:1,left:0}});
assert.equal(r.metrics.pixel_count,64);
assert.equal(r.metrics.rgb_mae_255,0);
near(r.metrics.ssim_luma_8x8,1);
assert.deepEqual(Array.from(new Uint8ClampedArray(r.diff).slice(320,324)),[0,0,0,255]);
// Equal composited RGB can have different alpha; report both independently.
a=pixels(1,1,[255,0,0,128]);b=pixels(1,1,[128,0,0,255]);
r=compare(a,b,1,1);
assert.equal(r.metrics.rgb_mae_255,0);
assert.equal(r.metrics.alpha_mae_255,127);
assert.equal(r.metrics.alpha_changed_pixels,1);
// Invisible RGB does not create a spurious visible difference.
r=compare(pixels(1,1,[255,255,255,0]),pixels(1,1,[0,0,0,0]),1,1);
assert.equal(r.metrics.rgb_mae_255,0);
// A change in just one channel contributes one third to RGB MAE.
r=compare(pixels(1,1,[0,0,0,255]),pixels(1,1,[30,0,0,255]),1,1);
near(r.metrics.rgb_mae_255,10);
near(r.metrics.rgb_rmse_255,Math.sqrt(300));
assert.equal(r.metrics.rgb_changed_pixels,1);
assert.match(compare(a,b,1,1,{margins:{top:1,right:0,bottom:0,left:0}}).error,/Margins/);
assert.match(compare(a,b,2,1).error,/dimensions/);
assert.match(compare(a,b,1,1,{threshold:-1}).error,/tolerance/);
assert.match(compare(a,b,1,1,{threshold:0.5}).error,/tolerance/);
console.log('Image Compare: numerical invariants and browser script syntax passed.');
