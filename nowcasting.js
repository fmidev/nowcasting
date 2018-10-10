var FCHOURS = ["0","1","2","3","4","5","6"]
var PARAMETERS = {"Paine":"Pressure", "Geopotentiaalikorkeus":"GeopHeight", "Lämpötila":"Temperature", "Kastepiste":"DewPoint", "Kosteus":"Humidity", "Tuulen nopeus":"WindSpeedMS", "Tuulen suunta":"WindDirection", "Tuuli u-komponentti":"WindUMS", "Tuuli v-komponentti":"WindVMS" }

$(function() {
    fillTables();
    selectParameters();
});

function fillTables() {
    var str='';for (var i in PARAMETERS){ str += '<option value="' + PARAMETERS[i] + '">' + i + '</option>'};
    $("#sel1").html(str);
    var str2='';for (var j in FCHOURS){ str2 += '<option value="' + FCHOURS[j] + '">' + FCHOURS[j] + '</option>'};
    $("#sel2").html(str2);
}

function selectParameters() {

    PARAMETER = $("#sel1 option:selected").val();
    FCHOUR = $("#sel2 option:selected").val();

    YEAR = STARTTIME.substr(0,4);
    MONTH = STARTTIME.substr(4,2);
    DAY = STARTTIME.substr(6,2);
    HOUR = STARTTIME.substr(8,2);
    MIN = STARTTIME.substr(10,2);
    SEC = STARTTIME.substr(12,2);

    var im1 = "/venison/cache/nowcasting/" + STARTTIME + "_" + ENDTIME + "_interp_adv_" + PARAMETER + "_fc=+" + FCHOUR + "h.png";
    var im2 = "/venison/cache/nowcasting/" + STARTTIME + "_" + ENDTIME + "_" + PARAMETER + "_colorbar.png";
    var label1 = PARAMETER + ' ' + YEAR + MONTH + DAY + ' ' + HOUR + ':' + MIN + ' ' + '+' + FCHOUR + 'h'

    var anim_links=[] ; for (var j in FCHOURS){ anim_links[j] = YEAR + MONTH + DAY + ' ' + HOUR + ':' + MIN + ' ' + '+' + FCHOURS[j] + 'h' };
    console.log(anim_links);
    var anim_files=[] ; for (var j in FCHOURS){ anim_files[j] = "/venison/cache/nowcasting/" + STARTTIME + "_" + ENDTIME + "_interp_adv_" + PARAMETER + "_fc=+" + FCHOURS[j] };
    console.log(anim_files);

    $("#data").attr("xlink:href", im1);
    $("#colorbar").attr("src", im2);

    document.getElementById('label').innerHTML = label1
    for (var j in anim_links) { document.getElementById('link_name_' + j).innerHTML = anim_links[j] }

}

function anim_image(idx) { 

    im1 = "/venison/cache/nowcasting/" + STARTTIME + "_" + ENDTIME + "_interp_adv_" + PARAMETER + "_fc=+" + idx + "h.png"; 
    $("#data").attr("xlink:href", im1);

    label1 = PARAMETER + ' ' + YEAR + MONTH + DAY + ' ' + HOUR + ':' + MIN + ' ' + '+' + idx + 'h'; 
    document.getElementById('label').innerHTML = label1

    document.getElementById("link_name_" + idx).style.color = "red";

}


function mouseOff(idx) {
    document.getElementById("link_name_" + idx).style.color = "black";
}



var anim = new JayDee.animator()
anim.frames = FCHOURS.length

anim.update = function(frame){

    if (typeof(frame)=='undefined')
        frame = this.frame
    
    im1 = "/venison/cache/nowcasting/" + STARTTIME + "_" + ENDTIME + "_interp_adv_" + PARAMETER + "_fc=+" + this.frame + "h.png"
    $("#data").attr("xlink:href", im1)

    label1 = PARAMETER + ' ' + YEAR + MONTH + DAY + ' ' + HOUR + ':' + MIN + ' ' + '+' + this.frame + 'h';
    document.getElementById('label').innerHTML = label1

    //document.getElementById("link_name_" + idx).style.color = "red";

}
