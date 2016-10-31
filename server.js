
var WORKERS = process.env.WEB_CONCURRENCY || 1;
var fs=require("fs");
var express = require('express');
var app = express();
var port = process.env.PORT || 8081;
var path = require('path');

app.use(express.static(path.join(__dirname,'')));
app.listen(port);
console.log("Express server listening on Port: "+port);