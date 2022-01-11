"use strict";
var templates = {};
var fs = require('fs');
var path = require('path');
var tplPath = path.join(path.dirname(__filename), '/views/');
function template(name) {
    if (templates[name]) {
        return templates[name];
    }
    templates[name] = fs.readFileSync("" + tplPath + name + ".html").toString();
    return templates[name];
}
exports.template = template;
//# sourceMappingURL=template.js.map