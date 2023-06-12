str='<img src="brown fox.jpg" title="The brown fox" />'
    +'<p>some text containing fox.</p>'
var word="fox";
word="(\\b"+ 
    word.replace(/([{}()[\]\\.?*+^$|=!:~-])/g, "\\$1")
    + "\\b)";
var r = new RegExp(word,"igm");
str.replace(/(>[^<]+<)/igm,function(a){
    return a.replace(r,"<span class='hl'>$1</span>");
});