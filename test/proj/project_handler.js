var fs = require('fs');
var path = require('path');

fs.readFile('./addons.txt', 'utf-8', function(err, data) {
    var sym = new String(data);
    let nus = parseInt(1);

    fs.writeFileSync('./addons.txt',  nus+sym);

    fs.readFile('./addons.txt', 'utf-8', function(err, data) {
        var dir = __dirname + "/project" + data;
        fs.mkdirSync(dir);
    });
});