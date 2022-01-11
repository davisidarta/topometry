var gulp = require('gulp');
var less = require('gulp-less');
var livereload = require('gulp-livereload');
gulp.task('less', function () {
    gulp.src('render/less/main.less')
        .pipe(less())
        .pipe(gulp.dest('render/css'))
        .pipe(livereload());
});
gulp.task('watch', function () {
    livereload.listen();
    gulp.watch('render/less/**/*.less', ['less']);
});
gulp.task('default', ['watch']);
//# sourceMappingURL=gulpfile.js.map