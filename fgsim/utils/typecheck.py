from typeguard import check_type


def istype(obj, objtype) -> bool:
    try:
        check_type("foo", obj, objtype)
        return True
    except TypeError:
        return False
