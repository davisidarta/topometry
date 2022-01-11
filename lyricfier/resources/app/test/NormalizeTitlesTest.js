"use strict";
var assert = require("assert");
var NormalizeTitles = require('../render/NormalizeTitles').NormalizeTitles;
var mm = new NormalizeTitles();
describe('NormalizeTitlesTest', function () {
    describe('#normalize()', function () {
        it('should eliminate - Remastered', function () {
            var testString = "Revolution - Remastered";
            var normalized = mm.normalize(testString);
            assert.equal('Revolution', normalized);
        });
        it('should eliminate - Remastered - YEAR', function () {
            var testString = "Revolution - Remastered 2015";
            var normalized = mm.normalize(testString);
            assert.equal('Revolution', normalized);
        });
        it('should eliminate - Remastered - YEAR', function () {
            var testString = "Revolution - Remastered 2015";
            var normalized = mm.normalize(testString);
            assert.equal('Revolution', normalized);
        });
        it('should eliminate - Remastered - Live / Remastered', function () {
            var testString = "Revolution - Live / Remastered";
            var normalized = mm.normalize(testString);
            assert.equal('Revolution', normalized);
        });
        it('should eliminate - Remastered - Live / Bonus Track', function () {
            var testString = "Revolution - Live / Bonus Track";
            var normalized = mm.normalize(testString);
            assert.equal('Revolution', normalized);
        });
        it('should eliminate - Mono / Remastered', function () {
            var testString = "Revolution - Mono / Remastered";
            var normalized = mm.normalize(testString);
            assert.equal('Revolution', normalized);
        });
    });
});
