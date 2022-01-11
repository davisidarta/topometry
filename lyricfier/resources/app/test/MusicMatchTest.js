"use strict";
var assert = require("assert");
var MusicMatch = require('../render/plugins/MusicMatch').MusicMatch;
describe('MusicMatch', function () {
    var mm = new MusicMatch();
    describe('#double-quote-escape()', function () {
        it('should return the body content with double quotes.', function () {
            var testString = "\"body\":\"My test \\\"string\\\"\",";
            var parsedString = mm.parseContent(testString);
            assert.equal('My test "string"', parsedString);
        });
        it('should return the body content with single quotes.', function () {
            var testString = "\"body\":\"My test 'string\\\"\",";
            var parsedString = mm.parseContent(testString);
            assert.equal("My test 'string\"", parsedString);
        });
    });
});
