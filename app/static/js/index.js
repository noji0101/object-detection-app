function loading(){
    $("#loading").show();
    $("#content").hide();       
}

function show_filename(){  
    var file = $('#input_img_file').prop('files')[0];
    $('#filename').text(file.name + '     is selected');
}