const { compiler } = require('gelorius-compiler');

compiler.createLanguage('.gristol', './syntax.json', './highlights.js');
