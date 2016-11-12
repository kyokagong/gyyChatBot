/**
 * Created by kyoka on 16/11/11.
 */

function runTensorflow() {
    var url = "/run_dgk/"
    $.ajax({
        url: url,
        async: true,
        success: function(ret){

        }
    });
    alert("success")
}

function getDgkLogs() {
    var ctx = document.getElementById("log")
    var url = "/get_dgk_logs/"
    $.ajax({
        url: url,
        async: false,
        success: function(ret){
            ret["logs"].forEach(function (log) {
                ctx.innerHTML+="<div>"+log+"</div>";
            })

        }
    });


}