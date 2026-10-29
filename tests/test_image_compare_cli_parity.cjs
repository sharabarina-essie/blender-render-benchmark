'use strict';
// Cross-language contract: Python CLI math must match the actual HTML worker.
// Usage: node tests/test_image_compare_cli_parity.cjs [python-executable]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const {spawnSync} = require('node:child_process');
const html = fs.readFileSync(path.join(__dirname, '../tools/image_compare.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const code = script.slice(script.indexOf('function comparePixels('), script.indexOf('\nconst $='));
let browser;
const scope = vm.createContext({self:{postMessage(value){browser=value;}}});
vm.runInContext(code, scope);
const python = process.argv[2] || 'python';
const bridge = `import json,sys,numpy as np
from compare_images import compare_arrays
p=json.load(sys.stdin)
a=np.array(p['a'],dtype=np.uint8).reshape(p['height'],p['width'],4)
b=np.array(p['b'],dtype=np.uint8).reshape(a.shape)
m,d=compare_arrays(a,b,margins=p['margins'],tolerance=p['threshold'],gain=p['gain'],make_difference=True)
print(json.dumps({'metrics':m,'diff':d.reshape(-1).tolist()},allow_nan=False))`;
let seed=1907;
function rand(){seed=(Math.imul(1664525,seed)+1013904223)>>>0;return seed>>>24;}
for (const [w,h,alpha,margins,threshold,gain] of [
 [1,1,true,{top:0,right:0,bottom:0,left:0},0,8],
 [9,9,false,{top:0,right:0,bottom:0,left:0},10,4],
 [31,25,true,{top:3,right:2,bottom:1,left:5},2,16],
 [128,96,true,{top:0,right:0,bottom:0,left:0},0,32],
 [19,17,false,{top:1,right:1,bottom:0,left:1},0,1],
]) {
 const a=Uint8ClampedArray.from({length:w*h*4},(_,i)=>i%4===3&&!alpha?255:rand());
 const b=a.slice();
 for(let i=0;i<b.length;i++)if(i%4!==3||alpha)b[i]=rand();
 scope.event={data:{width:w,height:h,a:a.buffer,b:b.buffer,margins,threshold,gain}};
 vm.runInContext('comparePixels(event)',scope);
 assert.ok(!browser.error,browser.error);
 const run=spawnSync(python,['-c',bridge],{cwd:path.join(__dirname,'..'),encoding:'utf8',input:JSON.stringify({width:w,height:h,a:Array.from(a),b:Array.from(b),margins,threshold,gain}),maxBuffer:10*1024*1024});
 assert.equal(run.status,0,run.stderr||String(run.error));
 const cli=JSON.parse(run.stdout);
 assert.deepEqual(Object.keys(cli.metrics).sort(),Object.keys(browser.metrics).sort());
 for(const key of Object.keys(cli.metrics)) {
  const actual=cli.metrics[key],expected=browser.metrics[key];
  if(typeof actual==='number' && !Number.isInteger(actual))assert.ok(Math.abs(actual-expected)<1e-11,`${w}x${h} ${key}: ${actual} vs ${expected}`);
  else assert.equal(actual,expected,`${w}x${h} ${key}`);
 }
 assert.deepEqual(cli.diff,Array.from(new Uint8ClampedArray(browser.diff)));
}
console.log('Python/browser parity passed: every metric within 1e-11 and every difference byte exact, including RGBA and partial ROI blocks.');
