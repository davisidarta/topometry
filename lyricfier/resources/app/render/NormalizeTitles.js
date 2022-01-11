"use strict";
var NormalizeTitles = (function () {
    function NormalizeTitles() {
    }
    NormalizeTitles.prototype.normalize = function (str) {
        var parts = str.split('-');
        if (parts.length === 2 && this.isDroppable(parts[1])) {
            return parts[0].trim();
        }
        return str;
    };
    NormalizeTitles.prototype.isDroppable = function (str) {
        return this.isRemastered(str) ||
            this.isBonusTrack(str) ||
            this.isLive(str);
    };
    NormalizeTitles.prototype.isRemastered = function (str) {
        return str.toLocaleLowerCase().indexOf('remastered') > -1;
    };
    NormalizeTitles.prototype.isBonusTrack = function (str) {
        return str.toLocaleLowerCase().indexOf('bonus track') > -1;
    };
    NormalizeTitles.prototype.isLive = function (str) {
        return str.toLocaleLowerCase().indexOf('live') > -1;
    };
    return NormalizeTitles;
}());
exports.NormalizeTitles = NormalizeTitles;
//# sourceMappingURL=NormalizeTitles.js.map